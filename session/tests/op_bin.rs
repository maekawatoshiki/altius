use altius_core::{dim::Dimensions, model::Model, node::Node, op::Op, tensor::Tensor};
use altius_session::interpreter::InterpreterSessionBuilder;

macro_rules! test_op {
    ($name:ident, $op:ident, $shape:expr) => {
        #[test]
        fn $name() {
            Tensor::seed_rng_from_u64(42);
            $op($shape.into())
        }
    };

    (! $name:ident, $op:ident, $shape:expr) => {
        #[test]
        #[should_panic]
        fn $name() {
            Tensor::seed_rng_from_u64(42);
            $op($shape.into())
        }
    };
}

macro_rules! op {
    ($name:ident, $op:ident, $bin:tt) => {
        #[cfg(test)]
        fn $name(shape: Dimensions) {
            let mut model = Model::default();
            let x = model.values.new_val_named("x");
            let y = model.values.new_val_named("y");
            let z = model.values.new_val_named("z");

            model.add_node(Node::new(Op::$op).with_ins(vec![x, y]).with_out(z));
            model.inputs.push(x);
            model.inputs.push(y);
            model.outputs.push(z);

            let sess = InterpreterSessionBuilder::new(&model).build().unwrap();
            let x_val = Tensor::rand::<f32>(shape.to_owned());
            let y_val = Tensor::rand::<f32>(shape);

            let expected = x_val
                .data::<f32>()
                .iter()
                .zip(y_val.data::<f32>().iter())
                .map(|(&x, &y)| x $bin y)
                .collect::<Vec<_>>();
            let actual = sess.run(vec![(x, x_val), (y, y_val)]).unwrap();
            assert_eq!(actual.len(), 1);
            assert!(allclose(actual[0].data::<f32>(), expected.as_slice()),
                "actual: {:?} vs expected: {:?}",
                &actual[0].data::<f32>()[..10], &expected.as_slice()[..10]);
        }
    };
}

op!(op_add, Add, +);
op!(op_sub, Sub, -);
op!(op_mul, Mul, *);
op!(op_div, Div, /);

test_op!(test_op_add_1, op_add, vec![1, 2]);
test_op!(test_op_add_2, op_add, vec![3, 1, 10]);
test_op!(test_op_add_3, op_add, vec![128, 3, 224, 224]);

test_op!(test_op_sub_1, op_sub, vec![1, 2]);
test_op!(test_op_sub_2, op_sub, vec![3, 1, 10]);
test_op!(test_op_sub_3, op_sub, vec![128, 3, 224, 224]);

test_op!(test_op_mul_1, op_mul, vec![1, 2]);
test_op!(test_op_mul_2, op_mul, vec![3, 1, 10]);
test_op!(test_op_mul_3, op_mul, vec![128, 3, 224, 224]);

test_op!(test_op_div_1, op_div, vec![1, 2]);
test_op!(test_op_div_2, op_div, vec![3, 1, 10]);
test_op!(test_op_div_3, op_div, vec![128, 3, 224, 224]);

#[cfg(test)]
fn allclose(x: &[f32], y: &[f32]) -> bool {
    let atol = 1e-5;
    let rtol = 1e-5;

    if x.len() != y.len() {
        return false;
    }

    x.iter().zip(y.iter()).all(|(x, y)| {
        ((x - y).abs() <= (atol + rtol * y.abs()))
            || (x.is_infinite() && y.is_infinite() && x.is_sign_positive() == y.is_sign_positive())
    })
}
