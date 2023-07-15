use std::{fs, path::Path};

use prost::Message;
use thiserror::Error;

use crate::{
    dim::Dimension as Dim,
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
    for (vals, proto) in [
        (&model.inputs, &mut graph_proto.input),
        (&model.outputs, &mut graph_proto.output),
    ] {
        for &id in vals {
            let val = &model.values.inner()[id];
            let Some(TypedShape { dims, elem_ty }) = &val.shape else {
                return Err(ModelSaveError::NoGraphInputShape);
            };
            let elem_ty: DataType = (*elem_ty).into();

            let ty = TypeProto {
                denotation: Some("TENSOR".to_string()),
                value: Some(TensorType(type_proto::Tensor {
                    elem_type: Some(elem_ty as i32),
                    shape: Some(TensorShapeProto {
                        dim: dims[0..]
                            .iter()
                            .map(|d| Dimension {
                                denotation: None,
                                value: match d {
                                    Dim::Fixed(d) => Some(DimValue::DimValue(*d as i64)),
                                    Dim::Dynamic(d) => Some(DimValue::DimParam(d.clone())),
                                },
                            })
                            .collect::<Vec<_>>(),
                    }),
                })),
            };

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
        let mut node_proto = NodeProto {
            name: node.name.clone(),
            op_type: node.op.name().to_string().into(),
            ..Default::default()
        };

        for &input_id in &node.inputs {
            let input = &model.values.inner()[input_id];
            let name = input
                .name
                .clone()
                .unwrap_or_else(|| format!("value.{}", input_id.index()));
            node_proto.input.push(name);
        }

        for &output_id in &node.outputs {
            let output = &model.values.inner()[output_id];
            let name = output
                .name
                .clone()
                .unwrap_or_else(|| format!("value.{}", output_id.index()));
            node_proto.output.push(name);
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
