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
    #[error("Input shape is not provided")]
    NoInputShape,

    #[error("Output shape is not provided")]
    NoOutputShape,
}

pub fn save_onnx(model: &Model, path: impl AsRef<Path>) -> Result<(), ModelSaveError> {
    let mut model_proto = ModelProto::default();
    let mut buf = Vec::new();

    model_proto.graph = encode_graph(model)?.into();
    model_proto.encode(&mut buf).unwrap();

    println!("buf: {:?}", buf);
    fs::write(path, buf).unwrap();

    Ok(())
}

fn encode_graph(model: &Model) -> Result<GraphProto, ModelSaveError> {
    let mut graph_proto = GraphProto::default();

    for &input_id in &model.inputs {
        let input = &model.values.inner()[input_id];
        let Some(TypedShape { dims, elem_ty }) =
            &input.shape else { return Err(ModelSaveError::NoInputShape) };

        let mut ty = TypeProto::default();
        ty.denotation = Some("TENSOR".to_string());
        ty.value = Some(TensorType(type_proto::Tensor {
            elem_type: Some((*elem_ty).into(): DataType as i32),
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

        graph_proto.input.push(ValueInfoProto {
            name: input.name.clone(),
            r#type: ty.into(),
            doc_string: "".to_string().into(),
        });
    }

    for &output_id in &model.outputs {
        let output = &model.values.inner()[output_id];
        let Some(TypedShape { dims, elem_ty }) =
            &output.shape else { return Err(ModelSaveError::NoOutputShape) };

        let mut ty = TypeProto::default();
        ty.denotation = Some("TENSOR".to_string());
        ty.value = Some(TensorType(type_proto::Tensor {
            elem_type: Some((*elem_ty).into(): DataType as i32),
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

        graph_proto.output.push(ValueInfoProto {
            name: output.name.clone(),
            r#type: ty.into(),
            doc_string: "".to_string().into(),
        });
    }

    // fn f<T>(x: T, f: impl FnOnce(&mut T)) -> T {
    //     let mut x = x;
    //     f(&mut x);
    //     x
    // }
    // graph_proto.node = vec![f(NodeProto::default(), |x| {
    //     x.name = "Add.0".to_string().into();
    //     x.op_type = "Add".to_string().into();
    //     x.input.push("input".to_string().into());
    //     x.input.push("input".to_string().into());
    //     x.output.push("output".to_string().into());
    // })];

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
    use super::load::load_onnx;
    let model = load_onnx("../models/mobilenetv3.onnx").unwrap();
    let _ = save_onnx(&model, "/tmp/test.onnx"); // .unwrap();
}
