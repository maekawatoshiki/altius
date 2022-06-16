use prost::Message;
use rustc_hash::FxHashMap;
use std::{fs, io, mem::transmute, path::Path};
use thiserror::Error;

use crate::{
    dim::Dimensions,
    model::Model,
    node::{Conv2d, HardSigmoid, MaxPool, Node, Op},
    tensor::{Tensor, TensorData},
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

        match node.op_type() {
            "Conv" => {
                let kernel = Dimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape").unwrap().ints,
                );
                let strides =
                    Dimensions::from_i64(&get_attribute(&node.attribute, "strides").unwrap().ints);
                let _conv = Node::new(Op::Conv2d(Conv2d {
                    auto_pad: "SAME_UPPER".into(),
                    kernel_shape: kernel,
                    strides,
                    ..Default::default()
                }))
                .with_ins(inputs)
                .with_outs(outputs)
                .alloc(&mut model.nodes);
            }
            "Add" => {
                let _add = Node::new(Op::Add)
                    .with_ins(inputs)
                    .with_outs(outputs)
                    .alloc(&mut model.nodes);
            }
            "Relu" => {
                let _relu = Node::new(Op::ReLU)
                    .with_ins(inputs)
                    .with_outs(outputs)
                    .alloc(&mut model.nodes);
            }
            "MaxPool" => {
                let kernel = Dimensions::from_i64(
                    &get_attribute(&node.attribute, "kernel_shape").unwrap().ints,
                );
                let strides =
                    Dimensions::from_i64(&get_attribute(&node.attribute, "strides").unwrap().ints);
                let _maxpool = Node::new(Op::MaxPool(MaxPool {
                    kernel_shape: kernel,
                    strides,
                }))
                .with_ins(inputs)
                .with_outs(outputs)
                .alloc(&mut model.nodes);
            }
            "Reshape" => {
                let _reshape = Node::new(Op::Reshape)
                    .with_ins(inputs)
                    .with_outs(outputs)
                    .alloc(&mut model.nodes);
            }
            "MatMul" => {
                let _matmul = Node::new(Op::MatMul)
                    .with_ins(inputs)
                    .with_outs(outputs)
                    .alloc(&mut model.nodes);
            }
            "HardSigmoid" => {
                let _hardsigmoid = Node::new(Op::HardSigmoid(HardSigmoid {
                    alpha: get_attribute(&node.attribute, "alpha").map_or(0.2, |a| a.f()),
                    beta: get_attribute(&node.attribute, "beta").map_or(0.5, |a| a.f()),
                }))
                .with_ins(inputs)
                .with_outs(outputs)
                .alloc(&mut model.nodes);
            }
            "Mul" => {
                let _mul = Node::new(Op::Mul)
                    .with_ins(inputs)
                    .with_outs(outputs)
                    .alloc(&mut model.nodes);
            }
            "GlobalAveragePool" => {
                todo!()
            }
            "Flatten" => {
                todo!()
            }
            "Gemm" => {
                todo!()
            }
            op => todo!("{}", op),
        }
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
    let dims = tensor.dims.iter().map(|&x| x as usize).collect::<Vec<_>>();
    let data = match DataType::from_i32(tensor.data_type()).unwrap() {
        DataType::Float if tensor.raw_data().is_empty() => {
            TensorData::F32(tensor.float_data.clone())
        }
        DataType::Float => TensorData::F32(
            unsafe { transmute::<&[u8], &[f32]>(tensor.raw_data()) }[..tensor.raw_data().len() / 4]
                .to_vec(),
        ),
        DataType::Int64 if tensor.raw_data().is_empty() => {
            TensorData::I64(tensor.int64_data.clone())
        }
        DataType::Int64 => TensorData::I64(
            unsafe { transmute::<&[u8], &[i64]>(tensor.raw_data()) }[..tensor.raw_data().len() / 8]
                .to_vec(),
        ),
        _ => todo!(),
    };
    Tensor::new(dims.into()).with_data(data)
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
