use std::path::Path;

use altius_core::{
    graph::Graph,
    model::Model,
    node::Node,
    onnx::{load_onnx, save::save_onnx},
    op::Op,
    tensor::{Tensor, TensorElemType, TypedFixedShape},
};
use altius_session_clang::ClangSessionBuilder;
use ndarray::CowArray;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[test]
fn cpu_ops_bin() {
    env_logger::init();

    [Op::Add, Op::Sub, Op::Mul, Op::Div]
        .into_par_iter()
        .for_each(|op| {
            let path = tempfile::NamedTempFile::new().unwrap();
            let path = path.path();
            Exporter::new(path, op).export();

            let x_ = Tensor::rand_of_type(TensorElemType::F32, vec![4, 2].into());
            let y_ = Tensor::rand_of_type(TensorElemType::F32, vec![4, 2].into());

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
                .into_shape((4, 2))
                .unwrap()
                .into_dimensionality()
                .unwrap();
            let y = CowArray::from(y_.data::<f32>())
                .into_shape((4, 2))
                .unwrap()
                .into_dimensionality()
                .unwrap();
            let x = Value::from_array(sess.allocator(), &x).unwrap();
            let y = Value::from_array(sess.allocator(), &y).unwrap();
            let z = &sess.run(vec![x, y]).unwrap()[0];
            let z = z.try_extract::<f32>().unwrap();
            let ort_z = z.view();
            assert!(ort_z.shape() == &[4, 2]);

            let sess = ClangSessionBuilder::new(load_onnx(path).unwrap())
                .build()
                .unwrap();
            let altius_z = &sess.run(vec![x_, y_]).unwrap()[0];
            assert!(altius_z.dims().as_slice() == &[4, 2]);

            ort_z
                .as_slice()
                .unwrap()
                .iter()
                .zip(altius_z.data::<f32>())
                .for_each(|(ort, altius)| {
                    assert!((ort - altius).abs() < 1e-6);
                });
        })
}

#[cfg(test)]
struct Exporter<'a> {
    path: &'a Path,
    op: Op,
}

#[cfg(test)]
impl<'a> Exporter<'a> {
    fn new(path: &'a Path, op: Op) -> Self {
        Self { path, op }
    }

    fn export(self) {
        // TODO: We need a better interface for building models.
        let mut model = Model {
            graph: Graph::default(),
            opset_version: 12,
        };
        let x = model.graph.values.new_val_named_and_shaped(
            "x",
            TypedFixedShape::new(vec![4, 2].into(), TensorElemType::F32),
        );
        let y = model.graph.values.new_val_named_and_shaped(
            "y",
            TypedFixedShape::new(vec![4, 2].into(), TensorElemType::F32),
        );
        let z = model.graph.values.new_val_named_and_shaped(
            "z",
            TypedFixedShape::new(vec![4, 2].into(), TensorElemType::F32),
        );
        model
            .graph
            .nodes
            .alloc(Node::new(self.op).with_ins(vec![x, y]).with_outs(vec![z]));
        model.graph.inputs.push(x);
        model.graph.inputs.push(y);
        model.graph.outputs.push(z);
        save_onnx(&model, self.path).unwrap();
    }
}
