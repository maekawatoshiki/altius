use altius_core::{model::Model, tensor::Tensor};
use altius_interpreter::Interpreter;

fn main() {
    use rayon::prelude::*;
    use std::cmp::Ordering;
    let mnist = mnist();

    let test = include_str!("../../core/examples/MNIST_test.txt");
    let test_lines: Vec<&str> = test.split("\n").collect();
    let mut inputs = vec![];
    for line in test_lines {
        if line.is_empty() {
            continue;
        }
        let nums: Vec<&str> = line.split(",").collect();
        let expected: i32 = nums[0].parse().unwrap();
        let pixels: Tensor = Tensor::new(vec![1, 1, 28, 28].into()).with_data(
            nums[1..]
                .iter()
                .map(|s| s.parse::<f32>().unwrap() / 255.0)
                .collect::<Vec<_>>()
                .into(),
        );
        inputs.push((expected, pixels));
    }
    let correct: i32 = inputs
        .par_iter()
        .map(|(expected, input)| {
            let mut i = Interpreter::new(&mnist, input.clone());
            let v = i.run();
            let inferred = v
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            // println!("expected: {}, inferred: {}", expected, inferred);
            (*expected == inferred as i32) as i32
        })
        .sum();
    // for (_expected, input) in &inputs {
    //     for x in 0..28 {
    //         for y in 0..28 {
    //             let pixel = input.at(&[0, 0, x, y]);
    //             print!("{}", if pixel > 0.5 { '#' } else { ' ' });
    //         }
    //         println!();
    //     }
    //     // break;
    // }
    println!("accuracy: {}", correct as f32 / inputs.len() as f32);
}

fn mnist() -> Model {
    use altius_core::{node::*, tensor::*};
    let mut m = Model::new();
    let input = m.new(Node::Input);
    let conv_weight = m.new(
        Tensor::new(vec![8, 1, 5, 5].into())
            .with_data(
                include!("../../core/examples/conv1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let conv = m.new(
        Conv2d {
            input_dims: vec![1, 1, 28, 28].into(),
            weight_dims: vec![8, 1, 5, 5].into(),
            weight_node: Some(conv_weight),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            padding: vec![2, 2].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(input),
            ..Default::default()
        }
        .into(),
    );
    // m.input_node = Some(conv);
    let add_input_b = m.new(
        Tensor::new(vec![8, 1, 1].into())
            .with_data(
                include!("../../core/examples/add1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add = m.new(
        Add {
            input_a_dims: vec![1, 8, 28, 28].into(),
            input_b_dims: vec![8, 1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_a_node: Some(conv),
            input_b_node: Some(add_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu = m.new(
        Relu {
            input_dims: vec![1, 8, 28, 28].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(add),
            ..Default::default()
        }
        .into(),
    );
    let max_pool = m.new(
        MaxPool {
            input_dims: vec![1, 8, 28, 28].into(),
            kernel: vec![2, 2].into(),
            stride: vec![2, 2].into(),
            output_dims: vec![1, 8, 14, 14].into(),
            input_node: Some(relu),
            ..Default::default()
        }
        .into(),
    );
    let conv2_weight = m.new(
        Tensor::new(vec![16, 8, 5, 5].into())
            .with_data(
                include!("../../core/examples/conv2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let conv2 = m.new(
        Conv2d {
            input_dims: vec![1, 8, 14, 14].into(),
            weight_dims: vec![16, 8, 5, 5].into(),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            padding: vec![2, 2].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_node: Some(max_pool),
            weight_node: Some(conv2_weight),
            ..Default::default()
        }
        .into(),
    );
    let add2_input_b = m.new(
        Tensor::new(vec![16, 1, 1].into())
            .with_data(
                include!("../../core/examples/add2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add2 = m.new(
        Add {
            input_a_dims: vec![1, 16, 14, 14].into(),
            input_b_dims: vec![16, 1, 1].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_a_node: Some(conv2),
            input_b_node: Some(add2_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu2 = m.new(
        Relu {
            input_dims: vec![1, 16, 14, 14].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_node: Some(add2),
            ..Default::default()
        }
        .into(),
    );
    let max_pool2 = m.new(
        MaxPool {
            input_dims: vec![1, 16, 14, 14].into(),
            kernel: vec![3, 3].into(),
            stride: vec![3, 3].into(),
            output_dims: vec![1, 16, 4, 4].into(),
            input_node: Some(relu2),
            ..Default::default()
        }
        .into(),
    );
    let reshape = m.new(
        Reshape {
            input_dims: vec![1, 16, 4, 4].into(),
            output_dims: vec![1, 256].into(),
            input_node: Some(max_pool2),
            ..Default::default()
        }
        .into(),
    );
    let reshape2_input = m.new(
        Tensor::new(vec![16, 4, 4, 10].into())
            .with_data(
                include!("../../core/examples/reshape1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let reshape2 = m.new(
        Reshape {
            input_dims: vec![16, 4, 4, 10].into(),
            output_dims: vec![256, 10].into(),
            input_node: Some(reshape2_input),
            ..Default::default()
        }
        .into(),
    );
    let mat_mal = m.new(
        MatMul {
            input_a_dims: vec![1, 256].into(),
            input_b_dims: vec![256, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(reshape),
            input_b_node: Some(reshape2),
            ..Default::default()
        }
        .into(),
    );
    let add3_input_b = m.new(
        Tensor::new(vec![1, 10].into())
            .with_data(
                include!("../../core/examples/add3")
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add3 = m.new(
        Add {
            input_a_dims: vec![1, 10].into(),
            input_b_dims: vec![1, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(mat_mal),
            input_b_node: Some(add3_input_b),
            ..Default::default()
        }
        .into(),
    );
    m.output_node = Some(add3);
    m
}
