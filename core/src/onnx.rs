use prost::Message;
use rustc_hash::FxHashMap;
use std::{fs, io, path::Path};
use thiserror::Error;

use crate::{
    dim::Dimensions,
    model::Model,
    node::{
        Cast, Concat, Conv2d, Flatten, Gemm, HardSigmoid, LeakyReLU, MaxPool, Node, Op, ReduceMin,
        Resize, Squeeze, Transpose,
    },
    tensor::{Tensor, TensorElemType},
};

use tensor_proto::DataType;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Error, Debug)]
pub enum ModelLoadError {
    Io(#[from] io::Error),
    NoGraph,
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
        let val = *name_to_val
            .entry(x.name())
            .or_insert_with(|| model.values.new_val_named(x.name()));
        model.inputs.push(val);
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
