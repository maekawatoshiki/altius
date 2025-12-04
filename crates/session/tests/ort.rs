use altius_core::{
    graph::Graph,
    model::Model,
    node::Node,
    onnx::save::save_onnx,
    op::Op,
    tensor::{TensorElemType, TypedFixedShape},
};
use ort::{session::Session, value::Tensor};

#[test]
fn ort_add() {
    let path = tempfile::NamedTempFile::new().unwrap();
    export_onnx(path.path().to_str().unwrap());

    let mut sess = Session::builder().unwrap().commit_from_file(path).unwrap();
    let x =
        Tensor::from_array(([4, 2], vec![1.0f32, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])).unwrap();
    let y =
        Tensor::from_array(([4, 2], vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])).unwrap();
    let z = &sess.run(ort::inputs![x, y]).unwrap()[0];
    let z = z.try_extract_array::<f32>().unwrap();

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
