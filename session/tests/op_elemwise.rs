use altius_core::{
    dim::Dimensions,
    model::Model,
    node::{Node, Op},
    tensor::Tensor,
};
use altius_session::interpreter::Interpreter;

#[test]
fn test_op_add() {
    op_add(vec![1, 2].into());
    op_add(vec![3, 1, 10].into());
    op_add(vec![128, 3, 224, 224].into());
}

fn op_add(shape: Dimensions) {
    let mut model = Model::default();
    let x = model.values.new_val_named("x");
    let y = model.values.new_val_named("y");
    let z = model.values.new_val_named("z");

    model.add_node(Node::new(Op::Add).with_ins(vec![x, y]).with_out(z));
    model.inputs.push(x);
    model.inputs.push(y);
    model.outputs.push(z);

    let sess = Interpreter::new(&model);
    let x_val = Tensor::rand::<f32>(shape.to_owned());
    let y_val = Tensor::rand::<f32>(shape);

    let expected = x_val
        .data::<f32>()
        .iter()
        .zip(y_val.data::<f32>().iter())
        .map(|(&x, &y)| x + y)
        .collect::<Vec<_>>();
    let actual = sess.run(vec![(x, x_val), (y, y_val)]).unwrap();
    assert_eq!(actual.len(), 1);
    assert!(allclose(actual[0].data::<f32>(), expected.as_slice()));
}

#[cfg(test)]
fn allclose(x: &[f32], y: &[f32]) -> bool {
    let atol = 1e-5;
    let rtol = 1e-5;

    if x.len() != y.len() {
        return false;
    }

    x.iter()
        .zip(y.iter())
        .all(|(x, y)| (x - y).abs() <= (atol + rtol * y.abs()))
}
