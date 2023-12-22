use std::{fs, path::Path};

use prost::Message;
use thiserror::Error;

use crate::{
    dim::Dimension as Dim,
    model::Model,
    op::Op,
    tensor::{TensorElemType, TypedShape},
};

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

use tensor_proto::DataType;
use tensor_shape_proto::{dimension::Value as DimValue, Dimension};
use type_proto::Value::TensorType;

use self::attribute_proto::AttributeType;

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

    #[error("Unknown opset version: {0}")]
    UnknownOpsetVersion(i64),
}

pub fn save_onnx(model: &Model, path: impl AsRef<Path>) -> Result<(), ModelSaveError> {
    fn opset_to_ir_version(opset: i64) -> Result<i64, ModelSaveError> {
        match opset {
            1..=8 => Ok(3),
            9 => Ok(4),
            10 => Ok(5),
            11 => Ok(6),
            12..=14 => Ok(7),
            15..=18 => Ok(8),
            19..=20 => Ok(9),
            _ => Err(ModelSaveError::UnknownOpsetVersion(opset)),
        }
    }

    let mut model_proto = ModelProto::default();
    let mut buf = Vec::new();

    model_proto.graph = encode_graph(model)?.into();
    model_proto.opset_import.push(OperatorSetIdProto {
        domain: Some("ai.onnx".to_string()),
        version: Some(model.opset_version),
    });
    model_proto.ir_version = Some(opset_to_ir_version(model.opset_version)?);
    model_proto.encode(&mut buf).unwrap();

    fs::write(path, buf).unwrap();

    Ok(())
}

fn encode_graph(model: &Model) -> Result<GraphProto, ModelSaveError> {
    let mut graph_proto = GraphProto::default();

    // Encode graph initializers.
    for (&id, tensor) in &model.graph.inits {
        let val = &model.graph.values.inner()[id];
        let Some(TypedShape { dims, elem_ty }) = &val.shape else {
            return Err(ModelSaveError::NoGraphInputShape);
        };
        let elem_ty: DataType = (*elem_ty).into();
        let name = val
            .name
            .clone()
            .unwrap_or_else(|| format!("initializer.{}", id.index()));
        assert_eq!(elem_ty, DataType::Float, "Only float is supported for now");

        graph_proto.initializer.push(TensorProto {
            name: Some(name),
            data_type: Some(elem_ty as i32),
            dims: dims[0..]
                .iter()
                .map(|d| match d {
                    Dim::Fixed(d) => *d as i64,
                    Dim::Dynamic(_d) => todo!(),
                })
                .collect::<Vec<_>>(),
            float_data: tensor.data::<f32>().iter().copied().collect::<Vec<_>>(),
            ..Default::default()
        });
    }

    // Encode graph inputs and outputs.
    for (vals, proto) in [
        (&model.graph.inputs, &mut graph_proto.input),
        (&model.graph.outputs, &mut graph_proto.output),
    ] {
        for &id in vals {
            let val = &model.graph.values.inner()[id];
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

    // TODO: We need to cover all ops.
    fn attrs(op: &Op) -> Vec<AttributeProto> {
        let mut attrs = vec![];
        match op {
            Op::Conv2d(c) => {
                attrs.push(AttributeProto {
                    name: "auto_pad".to_string().into(),
                    s: "SAME_UPPER".to_string().into_bytes().into(),
                    r#type: Some(AttributeType::String as i32),
                    ..Default::default()
                });
                attrs.push(AttributeProto {
                    name: "kernel_shape".to_string().into(),
                    ints: c.kernel_shape.iter().map(|x| *x as i64).collect::<Vec<_>>(),
                    r#type: Some(AttributeType::Ints as i32),
                    ..Default::default()
                });
            }
            _ => {}
        };
        attrs
    }

    fn op_type(name: &'static str) -> &'static str {
        match name {
            "Conv2d" => "Conv",
            name => name,
        }
    }

    // Encode nodes.
    for &node_id in &model.topo_sort_nodes() {
        let node = &model.graph.nodes[node_id];
        let mut node_proto = NodeProto {
            name: node.name.clone(),
            op_type: op_type(node.op.name()).to_string().into(),
            attribute: attrs(&node.op),
            ..Default::default()
        };

        for &input_id in &node.inputs {
            let input = &model.graph.values.inner()[input_id];
            let name = input
                .name
                .clone()
                .unwrap_or_else(|| format!("value.{}", input_id.index()));
            node_proto.input.push(name);
        }

        for &output_id in &node.outputs {
            let output = &model.graph.values.inner()[output_id];
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
    let model = load_onnx("../../models/mobilenetv3.onnx").unwrap();
    let _ = save_onnx(&model, "/tmp/test.onnx").unwrap();
    insta::assert_debug_snapshot!(load_onnx_model_proto("/tmp/test.onnx").unwrap());
}
