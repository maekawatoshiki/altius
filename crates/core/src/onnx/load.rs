use prost::{DecodeError, Message};
use rustc_hash::FxHashMap;
use std::{borrow::Cow, collections::hash_map::Entry, fs, io, path::Path};
use thiserror::Error;

use crate::{
    dim::Dimension,
    fixed_dim::FixedDimensions,
    model::Model,
    node::Node,
    op::{
        BatchNormalization, Cast, Concat, Constant, Conv2d, Flatten, Gather, Gemm, HardSigmoid,
        LayerNormalization, LeakyReLU, MaxPool, Op, ReduceMax, ReduceMean, ReduceMin, Resize,
        Shape, Softmax, Split, Squeeze, Transpose, Unsqueeze,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
};

use tensor_proto::DataType;
use tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use type_proto::Value::TensorType;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("{0}")]
    Io(#[from] io::Error),

    #[error("Model does not contain any graph")]
    NoGraph,

    #[error("Model is invalid: {0}")]
    InvalidModel(DecodeError),

    #[error("Model contains duplicated opsets")]
    DuplicateOpset,

    #[error("Value type is not specified")]
    NoValueType,

    #[error("Value shape is not specified")]
    NoValueShape,

    #[error("Attribute '{0}' is not specified")]
    NoAttribute(&'static str),

    #[error("Model contains unknown opset")]
    UnknownOpsetVersion,

    #[error("Something went wrong: {0}")]
    Todo(Cow<'static, str>),
}

pub fn load_onnx(path: impl AsRef<Path>) -> Result<Model, ModelLoadError> {
    let model_proto = load_onnx_model_proto(path)?;
    load_onnx_from_model_proto(model_proto)
}

pub fn load_onnx_from_buffer(buf: &[u8]) -> Result<Model, ModelLoadError> {
    let model = ModelProto::decode(buf).map_err(ModelLoadError::InvalidModel)?;
    load_onnx_from_model_proto(model)
}

pub fn load_onnx_from_model_proto(model_proto: ModelProto) -> Result<Model, ModelLoadError> {
    let graph = model_proto.graph.ok_or(ModelLoadError::NoGraph)?;
    let mut model = Model::default();
    let mut name_to_val = FxHashMap::default();

    let mut opset_version = None;
    for opset_import in &model_proto.opset_import {
        match opset_import.domain() {
            "" | "ai.onnx" if opset_version.is_none() => {
                opset_version = Some(opset_import.version())
            }
            "" | "ai.onnx" => return Err(ModelLoadError::DuplicateOpset),
            domain => {
                return Err(ModelLoadError::Todo(
                    format!("Custom domain ('{domain}') not supported yet").into(),
                ))
            }
        }
    }
    model.opset_version = opset_version.ok_or(ModelLoadError::DuplicateOpset)?;

    // Load initializers.
    for init in graph.initializer.iter() {
        let tensor = get_tensor(init)?;
        let val = *name_to_val
            .entry(init.name())
            .or_insert_with(|| model.graph.values.new_val_named(init.name()));
        model.graph.inits.insert(val, tensor);
    }

    // Load inputs.
    for (vals, vec) in [
        (&graph.input, &mut model.graph.inputs),
        (&graph.output, &mut model.graph.outputs),
    ] {
        for x in vals {
            let TensorType(tensor) = x
                .r#type
                .as_ref()
                .ok_or_else(|| ModelLoadError::NoValueType)?
                .value
                .as_ref()
                .ok_or_else(|| ModelLoadError::NoValueType)?
            else {
                return Err(ModelLoadError::Todo(
                    "Graph input must be tensor type".into(),
                ));
            };

            let dims: Vec<Dimension> = tensor
                .shape
                .as_ref()
                .ok_or_else(|| ModelLoadError::NoValueShape)?
                .dim
                .iter()
                .map(|d| match d.value.as_ref().unwrap() {
                    DimValue(i) => Dimension::Fixed(*i as usize),
                    DimParam(s) => Dimension::Dynamic(s.clone()),
                })
                .collect();

            let input = match name_to_val.entry(x.name()) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => *v.insert(model.graph.values.new_val_named_and_shaped(
                    x.name(),
                    TypedShape::new(
                        dims.into(),
                        DataType::from_i32(tensor.elem_type()).unwrap().try_into()?,
                    ),
                )),
            };

            vec.push(input);
        }
    }

    // Remove initializers from inputs if needed.
    model
        .graph
        .inputs
        .retain(|&x| !model.graph.inits.contains_key(&x));

    // Load nodes.
    for node in graph.node.iter() {
        let inputs = node
            .input
            .iter()
            .map(|input| {
                *name_to_val
                    .entry(input)
                    .or_insert_with(|| model.graph.values.new_val_named(input))
            })
            .collect();
        let outputs = node
            .output
            .iter()
            .map(|input| {
                *name_to_val
                    .entry(input)
                    .or_insert_with(|| model.graph.values.new_val_named(input))
            })
            .collect();

        let op = match node.op_type() {
            "Add" => Op::Add,
            "Sub" => Op::Sub,
            "Mul" => Op::Mul,
            "Div" => Op::Div,
            "Greater" => Op::Greater,
            "Pow" => Op::Pow,
            "Sqrt" => Op::Sqrt,
            "Relu" => Op::ReLU,
            "Sigmoid" => Op::Sigmoid,
            "Erf" => Op::Erf,
            "Tanh" => Op::Tanh,
            "Where" => Op::Where,
            "Softmax" => Op::Softmax(Softmax {
                axis: get_attribute(&node.attribute, "axis").map_or(-1, |a| a.i()),
            }),
            "Round" => Op::Round,
            "Exp" => Op::Exp,
            "Expand" => Op::Expand,
            "Range" => Op::Range,
            "Tile" => Op::Tile,
            "Split" => Op::Split(Split {
                axis: get_attribute(&node.attribute, "axis").map_or(0, |a| a.i()),
                split: get_attribute(&node.attribute, "split")
                    .map_or_else(|_| Vec::new(), |a| a.ints.clone()),
            }),
            "Slice" => Op::Slice,
            "Gather" => Op::Gather(Gather {
                axis: get_attribute(&node.attribute, "axis").map_or(0, |a| a.i()),
            }),
            "NonMaxSuppression" => Op::NonMaxSuppression,
            "Reshape" => Op::Reshape,
            "MatMul" => Op::MatMul,
            "GlobalAveragePool" => Op::GlobalAveragePool,
            "Conv" => {
                let auto_pad = get_attribute(&node.attribute, "auto_pad")
                    .map_or("NOTSET".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                let kernel_shape = FixedDimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape")?.ints,
                );
                let strides = get_attribute(&node.attribute, "strides")
                    .map_or(vec![1, 1].into(), |a| FixedDimensions::from_i64(&a.ints));
                let padding = get_attribute(&node.attribute, "pads")
                    .map_or(vec![0, 0].into(), |a| FixedDimensions::from_i64(&a.ints));
                let dilations = get_attribute(&node.attribute, "dilations")
                    .map_or(vec![1, 1].into(), |a| FixedDimensions::from_i64(&a.ints));
                let group = get_attribute(&node.attribute, "group").map_or(1, |a| a.i());
                Op::Conv2d(Conv2d {
                    auto_pad,
                    dilations,
                    kernel_shape,
                    strides,
                    group,
                    padding,
                    activation: None,
                })
            }
            "LeakyRelu" => Op::LeakyReLU(LeakyReLU {
                alpha: get_attribute(&node.attribute, "alpha").map_or(0.01, |a| a.f()),
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
                axis: get_attribute(&node.attribute, "axis")?.i(),
            }),
            "Transpose" => Op::Transpose(Transpose {
                perm: get_attribute(&node.attribute, "perm")?.ints.clone(),
            }),
            "Squeeze" if opset_version < Some(12) => Op::Squeeze(Squeeze {
                axes: get_attribute(&node.attribute, "axes")?.ints.clone(),
            }),
            "Squeeze" => Op::Squeeze(Squeeze { axes: vec![] }),
            "Unsqueeze" => Op::Unsqueeze(Unsqueeze {
                axes: get_attribute(&node.attribute, "axes")
                    .map_or_else(|_| Vec::new(), |a| a.ints.clone()),
            }),
            "ReduceMin" => Op::ReduceMin(ReduceMin {
                axes: get_attribute(&node.attribute, "axes")?.ints.clone(),
                keep_dims: get_attribute(&node.attribute, "keepdims").map_or(true, |a| a.i() != 0),
            }),
            "ReduceMax" => Op::ReduceMax(ReduceMax {
                axes: get_attribute(&node.attribute, "axes").map_or(vec![], |a| a.ints.clone()),
                keep_dims: get_attribute(&node.attribute, "keepdims").map_or(true, |a| a.i() != 0),
            }),
            "ReduceMean" => Op::ReduceMean(ReduceMean {
                axes: get_attribute(&node.attribute, "axes")?.ints.clone(),
                keep_dims: get_attribute(&node.attribute, "keepdims").map_or(true, |a| a.i() != 0),
            }),
            "Loop" => {
                // TODO
                let _body = get_attribute(&node.attribute, "body")?;
                log::warn!("Ignore loop body!");
                Op::Loop
            }
            "Cast" => {
                let to = TensorElemType::try_from(
                    DataType::from_i32(get_attribute(&node.attribute, "to")?.i() as i32)
                        .expect("Invalid ONNX"),
                )?;
                Op::Cast(Cast { to })
            }
            "MaxPool" => {
                let auto_pad = get_attribute(&node.attribute, "auto_pad")
                    .map_or("NOTSET".to_string(), |a| {
                        unsafe { std::str::from_utf8_unchecked(a.s()) }.to_string()
                    });
                let padding = get_attribute(&node.attribute, "pads")
                    .map_or(vec![0, 0].into(), |a| FixedDimensions::from_i64(&a.ints));
                let kernel = FixedDimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape")?.ints,
                );
                let strides =
                    FixedDimensions::from_i64(&get_attribute(&node.attribute, "strides")?.ints);
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
            "LayerNormalization" => Op::LayerNormalization(LayerNormalization {
                axis: get_attribute(&node.attribute, "axis").map_or(-1, |a| a.i()),
                stash_type: get_attribute(&node.attribute, "stash_type").map_or(1, |a| a.i()),
                epsilon: get_attribute(&node.attribute, "epsilon").map_or(1e-5, |a| a.f()),
            }),
            "Clip" => Op::Clip,
            "Shape" => Op::Shape(Shape {
                end: get_attribute(&node.attribute, "end").map_or(None, |a| a.i),
                start: get_attribute(&node.attribute, "start").map_or(0, |a| a.i()),
            }),
            "Constant" => Op::Constant(Constant {
                value: get_tensor(get_attribute(&node.attribute, "value").map_or_else(
                    |_| {
                        Err(ModelLoadError::Todo(
                            "Constant.value must be specified for now".into(),
                        ))
                    },
                    |a| Ok(a.t.as_ref().unwrap()),
                )?)?,
            }),
            op => return Err(ModelLoadError::Todo(format!("Unsupported op: {op}").into())),
        };

        model.graph.add_node(
            Node::new(op)
                .with_name(node.name.to_owned())
                .with_ins(inputs)
                .with_outs(outputs),
        );
    }

    Ok(model)
}

fn get_attribute<'a>(
    attrs: &'a [AttributeProto],
    name: &'static str,
) -> Result<&'a AttributeProto, ModelLoadError> {
    attrs
        .iter()
        .find(|x| x.name() == name)
        .ok_or_else(|| ModelLoadError::NoAttribute(name))
}

fn get_tensor(tensor: &TensorProto) -> Result<Tensor, ModelLoadError> {
    Ok(match DataType::from_i32(tensor.data_type()).unwrap() {
        DataType::Float if tensor.raw_data().is_empty() => Tensor::new(
            FixedDimensions::from_i64(&tensor.dims),
            tensor.float_data.clone(),
        ),
        DataType::Float => Tensor::new_from_raw(
            FixedDimensions::from_i64(&tensor.dims),
            TensorElemType::F32,
            tensor.raw_data().to_vec(),
        ),
        DataType::Int64 if tensor.raw_data().is_empty() => Tensor::new(
            FixedDimensions::from_i64(&tensor.dims),
            tensor.int64_data.clone(),
        ),
        DataType::Int64 => Tensor::new_from_raw(
            FixedDimensions::from_i64(&tensor.dims),
            TensorElemType::I64,
            tensor.raw_data().to_vec(),
        ),
        DataType::Int32 if tensor.raw_data().is_empty() => Tensor::new(
            FixedDimensions::from_i64(&tensor.dims),
            tensor.int32_data.clone(),
        ),
        DataType::Int32 => Tensor::new_from_raw(
            FixedDimensions::from_i64(&tensor.dims),
            TensorElemType::I32,
            tensor.raw_data().to_vec(),
        ),
        DataType::Bool => Tensor::new_from_raw(
            FixedDimensions::from_i64(&tensor.dims),
            TensorElemType::Bool,
            tensor.raw_data().to_vec(),
        ),
        t => {
            return Err(ModelLoadError::Todo(
                format!("Unsupported data type for tensor: {t:?}").into(),
            ))
        }
    })
}

impl TryFrom<DataType> for TensorElemType {
    type Error = ModelLoadError;

    fn try_from(ty: DataType) -> Result<Self, Self::Error> {
        match ty {
            DataType::Bool => Ok(TensorElemType::Bool),
            DataType::Int32 => Ok(TensorElemType::I32),
            DataType::Int64 => Ok(TensorElemType::I64),
            DataType::Float => Ok(TensorElemType::F32),
            ty => Err(ModelLoadError::Todo(
                format!("Unsupported tensor element type: {ty:?}").into(),
            )),
        }
    }
}

pub fn load_onnx_model_proto(path: impl AsRef<Path>) -> Result<ModelProto, io::Error> {
    let model = ModelProto::decode(&*fs::read(path)?)?;
    Ok(model)
}

#[test]
fn load_mnist_proto() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join("mnist-8.onnx");
    let model = load_onnx_model_proto(model_path).unwrap();
    assert!(model.graph.is_some());
}

#[test]
fn load_mnist() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join("mnist-8.onnx");
    let model = load_onnx(model_path).unwrap();
    println!("{:#?}", model.graph.nodes);
    println!("{:#?}", model.graph.inputs);
    println!("{:#?}", model.graph.outputs);
}
