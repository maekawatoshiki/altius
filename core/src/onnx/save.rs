use std::{fs, path::Path};

use prost::Message;
use thiserror::Error;

use crate::{
    model::Model,
    tensor::{TensorElemType, TypedShape},
};

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

use tensor_proto::DataType;
use tensor_shape_proto::{dimension::Value as DimValue, Dimension};
use type_proto::Value::TensorType;

#[derive(Error, Debug)]
pub enum ModelSaveError {
    // #[error("{0}")]
    // Io(#[from] io::Error),
    //
    // #[error("Model does not contain any graph")]
    // NoGraph,
    //
    // #[error("Model contains duplicated opsets")]
    // DuplicateOpset,
    //
    // #[error("Model contains unknown opset")]
    // UnknownOpsetVersion,
    //
    // #[error("Something went wrong: {0}")]
    // Todo(Cow<'static, str>),
    #[error("Graph input shape is not provided")]
    NoGraphInputShape,

    #[error("Graph output shape is not provided")]
    NoGraphOutputShape,

    #[error("Node input name is not provided")]
    NoNodeInputName,

    #[error("Node output name is not provided")]
    NoNodeOutputName,
}

pub fn save_onnx(model: &Model, path: impl AsRef<Path>) -> Result<(), ModelSaveError> {
    let mut model_proto = ModelProto::default();
    let mut buf = Vec::new();

    model_proto.graph = encode_graph(model)?.into();
    model_proto.encode(&mut buf).unwrap();

    fs::write(path, buf).unwrap();

    Ok(())
}

fn encode_graph(model: &Model) -> Result<GraphProto, ModelSaveError> {
    let mut graph_proto = GraphProto::default();

    // Encode graph inputs and outputs.
    for (vals, proto) in vec![
        (&model.inputs, &mut graph_proto.input),
        (&model.outputs, &mut graph_proto.output),
    ] {
        for &id in vals {
            let val = &model.values.inner()[id];
            let Some(TypedShape { dims, elem_ty }) =
                &val.shape else { return Err(ModelSaveError::NoGraphInputShape) };
            let elem_ty: DataType = (*elem_ty).into();

            let mut ty = TypeProto::default();
            ty.denotation = Some("TENSOR".to_string());
            ty.value = Some(TensorType(type_proto::Tensor {
                elem_type: Some(elem_ty as i32),
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| Dimension {
                            denotation: None,
                            value: Some(DimValue::DimValue(*d as i64)),
                        })
                        .collect::<Vec<_>>(),
                }),
            }));

            proto.push(ValueInfoProto {
                name: val.name.clone(),
                r#type: ty.into(),
                doc_string: "".to_string().into(),
            });
        }
    }

    // Encode nodes.
    for &node_id in &model.topo_sort_nodes() {
        let node = &model.nodes[node_id];
        let mut node_proto = NodeProto::default();

        node_proto.name = node.name.clone();
        node_proto.op_type = node.op.name().to_string().into();

        for &input_id in &node.inputs {
            let input = &model.values.inner()[input_id];
            let Some(name) = &input.name
                else { return Err(ModelSaveError::NoNodeInputName); };
            node_proto.input.push(name.clone());
        }

        for &output_id in &node.outputs {
            let output = &model.values.inner()[output_id];
            let Some(name) = &output.name
                else { return Err(ModelSaveError::NoNodeOutputName); };
            node_proto.output.push(name.clone());
        }

        graph_proto.node.push(node_proto);
    }

    Ok(graph_proto)
}

impl From<TensorElemType> for DataType {
    fn from(ty: TensorElemType) -> Self {
        match ty {
            TensorElemType::F32 => DataType::Float,
            TensorElemType::I32 => DataType::Int32,
            TensorElemType::I64 => DataType::Int64,
            TensorElemType::Bool => DataType::Bool,
        }
    }
}

#[test]
fn test_save_onnx() {
    use super::load::{load_onnx, load_onnx_model_proto};
    let model = load_onnx("../models/mobilenetv3.onnx").unwrap();
    let _ = save_onnx(&model, "/tmp/test.onnx").unwrap();
    insta::assert_debug_snapshot!(load_onnx_model_proto("/tmp/test.onnx").unwrap());
}
