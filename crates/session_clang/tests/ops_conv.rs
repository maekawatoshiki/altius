use std::path::Path;

use altius_core::{
    graph::Graph,
    model::Model,
    node::Node,
    onnx::{load_onnx, save::save_onnx},
    op::{Conv2d, Op},
    tensor::{Tensor, TensorElemType, TypedFixedShape},
};
use altius_session_clang::ClangSessionBuilder;
use ndarray::CowArray;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};

#[test]
fn cpu_ops_conv() {
    env_logger::init();

    let path = tempfile::NamedTempFile::new().unwrap();
    let path = path.path();
    Exporter::new(path).export();

    let x_ = Tensor::rand_of_type(TensorElemType::F32, vec![1, 1, 28, 28].into());

    let env = Environment::builder()
        .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])
        .build()
        .unwrap()
        .into_arc();
    let sess = SessionBuilder::new(&env)
        .unwrap()
        .with_model_from_file(path)
        .unwrap();
    let x = CowArray::from(x_.data::<f32>())
        .into_shape((1, 1, 28, 28))
        .unwrap()
        .into_dimensionality()
        .unwrap();
    let x = Value::from_array(sess.allocator(), &x).unwrap();
    let z = &sess.run(vec![x]).unwrap()[0];
    let z = z.try_extract::<f32>().unwrap();
    let ort_z = z.view();
    assert!(ort_z.shape() == &[1, 8, 28, 28]);

    let sess = ClangSessionBuilder::new(load_onnx(path).unwrap())
        .build()
        .unwrap();
    let altius_z = &sess.run(vec![x_]).unwrap()[0];
    assert!(altius_z.dims().as_slice() == &[1, 8, 28, 28]);

    ort_z
        .as_slice()
        .unwrap()
        .iter()
        .zip(altius_z.data::<f32>())
        .for_each(|(ort, altius)| {
            assert!((ort - altius).abs() < 1e-5, "{} != {}", ort, altius);
        });
}

#[cfg(test)]
struct Exporter<'a> {
    path: &'a Path,
}

#[cfg(test)]
impl<'a> Exporter<'a> {
    fn new(path: &'a Path) -> Self {
        Self { path }
    }

    fn export(self) {
        // TODO: We need a better interface for building models.
        let mut model = Model {
            graph: Graph::default(),
            opset_version: 12,
        };
        let x = model.graph.values.new_val_named_and_shaped(
            "x",
            TypedFixedShape::new(vec![1, 1, 28, 28].into(), TensorElemType::F32),
        );
        let y = model.graph.values.new_val_named_and_shaped(
            "y",
            TypedFixedShape::new(vec![8, 1, 5, 5].into(), TensorElemType::F32),
        );
        let z = model.graph.values.new_val_named_and_shaped(
            "z",
            TypedFixedShape::new(vec![1, 8, 28, 28].into(), TensorElemType::F32),
        );
        model.graph.nodes.alloc(
            Node::new(Op::Conv2d(Conv2d {
                auto_pad: "SAME_UPPER".into(),
                kernel_shape: vec![5, 5].into(),
                strides: vec![1, 1].into(),
                group: 1,
                dilations: vec![1, 1].into(),
                ..Default::default()
            }))
            .with_ins(vec![x, y])
            .with_outs(vec![z]),
        );
        model.graph.inputs.push(x);
        model.graph.outputs.push(z);
        model.graph.inits.insert(
            y,
            Tensor::rand_of_type(TensorElemType::F32, vec![8, 1, 5, 5].into()),
        );
        save_onnx(&model, self.path).unwrap();
    }
}
