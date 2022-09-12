use prost::Message;
use rustc_hash::FxHashMap;
use std::{borrow::Cow, fs, io, path::Path};
use thiserror::Error;

use crate::{
    dim::Dimensions,
    model::Model,
    node::{
        BatchNormalization, Cast, Concat, Constant, Conv2d, Flatten, Gemm, HardSigmoid, LeakyReLU,
        MaxPool, Node, Op, ReduceMin, Resize, Shape, Squeeze, Transpose,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
};

use tensor_proto::DataType;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Error, Debug)]
pub enum ModelLoadError {
    Io(#[from] io::Error),
    NoGraph,
    Todo(Cow<'static, str>),
}

pub fn load_onnx(path: impl AsRef<Path>) -> Result<Model, ModelLoadError> {
    let model_proto = load_onnx_model_proto(path)?;
    let graph = model_proto.graph.ok_or(ModelLoadError::NoGraph)?;
    let mut model = Model::default();

    let mut name_to_val = FxHashMap::default();

    // Load initializers.
    for init in graph.initializer.iter() {
        let tensor = get_tensor(init);
        let val = *name_to_val
            .entry(init.name())
            .or_insert_with(|| model.values.new_val_named(init.name()));
        model.inits.insert(val, tensor);
    }

    // Load inputs.
    for x in graph.input.iter() {
        let val = x.r#type.as_ref().unwrap().value.as_ref().unwrap();
        let mut dims = vec![];
        let mut is_dynamic_shape = false;
        let tensor = if let type_proto::Value::TensorType(tensor) = val {
            tensor
        } else {
            return Err(ModelLoadError::Todo(
                "Graph input must be tensor type".into(),
            ));
        };
        for d in tensor
            .shape
            .as_ref()
            .unwrap()
            .dim
            .iter()
            .map(|d| d.value.as_ref().unwrap())
        {
            match d {
                tensor_shape_proto::dimension::Value::DimValue(i) => dims.push(*i),
                _ => {
                    is_dynamic_shape = true;
                    break;
                }
            }
        }
        let input = if is_dynamic_shape {
            *name_to_val
                .entry(x.name())
                .or_insert_with(|| model.values.new_val_named(x.name()))
        } else {
            *name_to_val.entry(x.name()).or_insert_with(|| {
                model.values.new_val_named_and_shaped(
                    x.name(),
                    TypedShape::new(
                        Dimensions::from_i64(&dims),
                        DataType::from_i32(tensor.elem_type()).unwrap().into(),
                    ),
                )
            })
        };
        model.inputs.push(input);
    }

    // Load outputs.
    for x in graph.output.iter() {
        let val = *name_to_val
            .entry(x.name())
            .or_insert_with(|| model.values.new_val_named(x.name()));
        model.outputs.push(val);
    }

    // Load nodes.
    for node in graph.node.iter() {
        let inputs = node
            .input
            .iter()
            .map(|input| {
                *name_to_val
                    .entry(input)
                    .or_insert_with(|| model.values.new_val_named(input))
            })
            .collect();
        let outputs = node
            .output
            .iter()
            .map(|input| {
                *name_to_val
                    .entry(input)
                    .or_insert_with(|| model.values.new_val_named(input))
            })
            .collect();

        let op = match node.op_type() {
            "Add" => Op::Add,
            "Sub" => Op::Sub,
            "Mul" => Op::Mul,
            "Div" => Op::Div,
            "Relu" => Op::ReLU,
            "Sigmoid" => Op::Sigmoid,
            "Round" => Op::Round,
            "Exp" => Op::Exp,
            "Tile" => Op::Tile,
            "Slice" => Op::Slice,
            "NonMaxSuppression" => Op::NonMaxSuppression,
            "Reshape" => Op::Reshape,
            "MatMul" => Op::MatMul,
            "GlobalAveragePool" => Op::GlobalAveragePool,
            "Conv" => {
                let auto_pad = get_attribute(&node.attribute, "auto_pad")
                    .map_or("NOTSET".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                let kernel_shape = Dimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape").unwrap().ints,
                );
                let strides =
                    Dimensions::from_i64(&get_attribute(&node.attribute, "strides").unwrap().ints);
                let padding = get_attribute(&node.attribute, "pads")
                    .map_or(vec![0, 0].into(), |a| Dimensions::from_i64(&a.ints));
                let group = get_attribute(&node.attribute, "group").map_or(1, |a| a.i());
                Op::Conv2d(Conv2d {
                    auto_pad,
                    kernel_shape,
                    strides,
                    group,
                    padding,
                })
            }
            "LeakyRelu" => Op::LeakyReLU(LeakyReLU {
                alpha: get_attribute(&node.attribute, "alpha").unwrap().f(),
            }),
            "Resize" => {
                let coordinate_transformation_mode =
                    get_attribute(&node.attribute, "coordinate_transformation_mode")
                        .map_or("half_pixel".to_string(), |a| {
                            unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                        });
                let cubic_coeff_a =
                    get_attribute(&node.attribute, "cubic_coeff_a").map_or(-0.75, |a| a.f());
                let exclude_outside =
                    get_attribute(&node.attribute, "exclude_outside").map_or(0, |a| a.i());
                let extrapolation_value =
                    get_attribute(&node.attribute, "extrapolation_value").map_or(0.0, |a| a.f());
                let mode = get_attribute(&node.attribute, "mode")
                    .map_or("nearest".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                let nearest_mode = get_attribute(&node.attribute, "nearest_mode")
                    .map_or("round_prefer_floor".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                Op::Resize(Resize {
                    coordinate_transformation_mode,
                    cubic_coeff_a,
                    exclude_outside,
                    extrapolation_value,
                    mode,
                    nearest_mode,
                })
            }
            "Concat" => Op::Concat(Concat {
                axis: get_attribute(&node.attribute, "axis").unwrap().i(),
            }),
            "Transpose" => Op::Transpose(Transpose {
                perm: get_attribute(&node.attribute, "perm").unwrap().ints.clone(),
            }),
            "Squeeze" => Op::Squeeze(Squeeze {
                axes: get_attribute(&node.attribute, "axes").unwrap().ints.clone(),
            }),
            "Unsqueeze" => Op::Squeeze(Squeeze {
                axes: get_attribute(&node.attribute, "axes").unwrap().ints.clone(),
            }),
            "ReduceMin" => Op::ReduceMin(ReduceMin {
                axes: get_attribute(&node.attribute, "axes").unwrap().ints.clone(),
                keep_dims: get_attribute(&node.attribute, "keepdims").map_or(1, |a| a.i()),
            }),
            "Loop" => {
                // TODO
                let _body = get_attribute(&node.attribute, "body").unwrap();
                log::debug!("Ignore loop body!");
                Op::Loop
            }
            "Cast" => {
                let to = match DataType::from_i32(
                    get_attribute(&node.attribute, "to").unwrap().i() as i32
                )
                .unwrap()
                {
                    DataType::Float => TensorElemType::F32,
                    DataType::Int32 => TensorElemType::I32,
                    e => todo!("Cast to {:?}", e),
                };
                Op::Cast(Cast { to })
            }
            "MaxPool" => {
                let auto_pad = get_attribute(&node.attribute, "auto_pad")
                    .map_or("NOTSET".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                let padding = get_attribute(&node.attribute, "pads")
                    .map_or(vec![0, 0].into(), |a| Dimensions::from_i64(&a.ints));
                let kernel = Dimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape").unwrap().ints,
                );
                let strides =
                    Dimensions::from_i64(&get_attribute(&node.attribute, "strides").unwrap().ints);
                Op::MaxPool(MaxPool {
                    auto_pad,
                    padding,
                    kernel_shape: kernel,
                    strides,
                })
            }
            "HardSigmoid" => Op::HardSigmoid(HardSigmoid {
                alpha: get_attribute(&node.attribute, "alpha").map_or(0.2, |a| a.f()),
                beta: get_attribute(&node.attribute, "beta").map_or(0.5, |a| a.f()),
            }),
            "Flatten" => Op::Flatten(Flatten {
                axis: get_attribute(&node.attribute, "axis").map_or(1, |a| a.i()),
            }),
            "Gemm" => Op::Gemm(Gemm {
                alpha: get_attribute(&node.attribute, "alpha").map_or(1.0, |a| a.f()),
                beta: get_attribute(&node.attribute, "beta").map_or(1.0, |a| a.f()),
                trans_a: get_attribute(&node.attribute, "transA").map_or(false, |a| a.i() == 1),
                trans_b: get_attribute(&node.attribute, "transB").map_or(false, |a| a.i() == 1),
            }),
            "BatchNormalization" => Op::BatchNormalization(BatchNormalization {
                epsilon: get_attribute(&node.attribute, "epsilon").map_or(1e-5, |a| a.f()),
                momentum: get_attribute(&node.attribute, "momentum").map_or(1e-5, |a| a.f()),
                training_mode: get_attribute(&node.attribute, "training_mode")
                    .map_or(false, |a| a.i() != 0),
            }),
            "Clip" => Op::Clip,
            "Shape" => Op::Shape(Shape {
                end: get_attribute(&node.attribute, "end").and_then(|a| a.i),
                start: get_attribute(&node.attribute, "start").map_or(0, |a| a.i()),
            }),
            "Constant" => Op::Constant(Constant {
                value: get_tensor(get_attribute(&node.attribute, "value").map_or_else(
                    || {
                        Err(ModelLoadError::Todo(
                            "Constant.value must be specified for now".into(),
                        ))
                    },
                    |a| Ok(a.t.as_ref().unwrap()),
                )?),
            }),
            op => todo!("op: {}", op),
        };

        let _ = Node::new(op)
            .with_ins(inputs)
            .with_outs(outputs)
            .alloc(&mut model.nodes);
    }

    Ok(model)
}

fn get_attribute<'a>(
    attrs: &'a [AttributeProto],
    name: &'static str,
) -> Option<&'a AttributeProto> {
    attrs.iter().find(|x| x.name() == name)
}

fn get_tensor(tensor: &TensorProto) -> Tensor {
    match DataType::from_i32(tensor.data_type()).unwrap() {
        DataType::Float if tensor.raw_data().is_empty() => Tensor::new(
            Dimensions::from_i64(&tensor.dims),
            tensor.float_data.clone(),
        ),
        DataType::Float => Tensor::new_from_raw(
            Dimensions::from_i64(&tensor.dims),
            TensorElemType::F32,
            tensor.raw_data().to_vec(),
        ),
        DataType::Int64 if tensor.raw_data().is_empty() => Tensor::new(
            Dimensions::from_i64(&tensor.dims),
            tensor.int64_data.clone(),
        ),
        DataType::Int64 => Tensor::new_from_raw(
            Dimensions::from_i64(&tensor.dims),
            TensorElemType::I64,
            tensor.raw_data().to_vec(),
        ),
        DataType::Int32 if tensor.raw_data().is_empty() => Tensor::new(
            Dimensions::from_i64(&tensor.dims),
            tensor.int32_data.clone(),
        ),
        DataType::Int32 => Tensor::new_from_raw(
            Dimensions::from_i64(&tensor.dims),
            TensorElemType::I32,
            tensor.raw_data().to_vec(),
        ),
        DataType::Bool => Tensor::new_from_raw(
            Dimensions::from_i64(&tensor.dims),
            TensorElemType::Bool,
            tensor.raw_data().to_vec(),
        ),
        e => todo!("data type: {e:?}"),
    }
}

impl From<DataType> for TensorElemType {
    fn from(ty: DataType) -> Self {
        match ty {
            DataType::Bool => TensorElemType::Bool,
            DataType::Int32 => TensorElemType::I32,
            DataType::Int64 => TensorElemType::I64,
            DataType::Float => TensorElemType::F32,
            _ => todo!("Unsupported tensor element type: {ty:?}"),
        }
    }
}

pub fn load_onnx_model_proto(path: impl AsRef<Path>) -> Result<ModelProto, io::Error> {
    let model = ModelProto::decode(&*fs::read(path)?)?;
    Ok(model)
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[test]
fn load_mnist_proto() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join("mnist-8.onnx");
    let model = load_onnx_model_proto(model_path).unwrap();
    assert!(model.graph.is_some());
}

#[test]
fn load_mnist() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join("mnist-8.onnx");
    let model = load_onnx(model_path).unwrap();
    println!("{:#?}", model.nodes);
    println!("{:#?}", model.inputs);
    println!("{:#?}", model.outputs);
}
