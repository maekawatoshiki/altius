use altius_core::{
    graph::Graph,
    model::Model,
    node::Node,
    onnx::save::save_onnx,
    op::Op,
    tensor::{TensorElemType, TypedFixedShape},
};
use ndarray::CowArray;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};

#[test]
fn ort_add() {
    let path = "/tmp/add.onnx";
    export_onnx(path);

    let env = Environment::builder()
        .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])
        .build()
        .unwrap()
        .into_arc();
    let sess = SessionBuilder::new(&env)
        .unwrap()
        .with_model_from_file(path)
        .unwrap();
    let x = CowArray::from(&[1.0f32, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
        .into_shape((4, 2))
        .unwrap()
        .into_dimensionality()
        .unwrap();
    let y = CowArray::from(&[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
        .into_shape((4, 2))
        .unwrap()
        .into_dimensionality()
        .unwrap();
    let x = Value::from_array(sess.allocator(), &x).unwrap();
    let y = Value::from_array(sess.allocator(), &y).unwrap();
    let z = &sess.run(vec![x, y]).unwrap()[0];
    let z = z.try_extract::<f32>().unwrap();
    let z = z.view();

    assert!(z.shape() == &[4, 2]);
    assert!(z.as_slice() == Some(&[3.0f32, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0]));
}

#[cfg(test)]
fn export_onnx(path: &str) {
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
        .alloc(Node::new(Op::Add).with_ins(vec![x, y]).with_outs(vec![z]));
    model.graph.inputs.push(x);
    model.graph.inputs.push(y);
    model.graph.outputs.push(z);
    save_onnx(&model, path).unwrap();
}
