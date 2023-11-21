use altius_core::{
    graph::Graph,
    model::Model,
    node::Node,
    onnx::{load_onnx, save::save_onnx},
    op::Op,
    tensor::{Tensor, TensorElemType, TypedFixedShape},
};
use altius_session_cpu::CPUSessionBuilder;
use ndarray::CowArray;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};

#[test]
fn cpu_ops_bin() {
    // env_logger::init();

    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        let path = tempfile::NamedTempFile::new().unwrap();
        let path = path.path();
        export_onnx(path.to_str().unwrap(), op);

        let x_ = vec![1.0f32, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let y_ = vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let env = Environment::builder()
            .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])
            .build()
            .unwrap()
            .into_arc();
        let sess = SessionBuilder::new(&env)
            .unwrap()
            .with_model_from_file(path)
            .unwrap();
        let x = CowArray::from(&x_)
            .into_shape((4, 2))
            .unwrap()
            .into_dimensionality()
            .unwrap();
        let y = CowArray::from(&y_)
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

        let sess = CPUSessionBuilder::new(load_onnx(path).unwrap())
            .build()
            .unwrap();
        let x = Tensor::new(vec![4, 2].into(), x_);
        let y = Tensor::new(vec![4, 2].into(), y_);
        let altius_z = &sess.run(vec![x, y]).unwrap()[0];
        assert!(altius_z.dims().as_slice() == &[4, 2]);

        ort_z
            .as_slice()
            .unwrap()
            .iter()
            .zip(altius_z.data::<f32>())
            .for_each(|(ort, altius)| {
                assert!((ort - altius).abs() < 1e-6);
            });
    }
}

#[cfg(test)]
fn export_onnx(path: &str, op: Op) {
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
        .alloc(Node::new(op).with_ins(vec![x, y]).with_outs(vec![z]));
    model.graph.inputs.push(x);
    model.graph.inputs.push(y);
    model.graph.outputs.push(z);
    save_onnx(&model, path).unwrap();
}
