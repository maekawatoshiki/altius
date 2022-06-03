use altius_core::model::Model2;
use altius_core::{model::Model, node::*, tensor::*};
use altius_interpreter::Interpreter;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::env::args;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::Path;

fn main() {
    let model_root = &args().collect::<Vec<String>>()[1];
    let _mnist = mnist2(model_root);
    let mnist = mnist(model_root);

    let mut inputs = vec![];
    for line in fs::read_to_string(Path::new(model_root).join("MNIST_test.txt"))
        .unwrap()
        .split("\n")
    {
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

    let mut correct: i32 = 0;
    let validation_count = 100;
    let repeat = 5;

    let start = ::std::time::Instant::now();

    for _ in 0..repeat {
        correct = inputs
            .par_iter()
            .take(validation_count)
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
                (*expected == inferred as i32) as i32
            })
            .sum();
    }

    let end = start.elapsed();
    println!("elapsed: {:?}", end);
    println!(
        "fps: {:?}",
        (validation_count as f64 * repeat as f64) / end.as_secs_f64()
    );

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

    println!("accuracy: {}", correct as f32 / validation_count as f32);
}

fn read_f32_list_from<P: AsRef<Path>>(filename: P) -> Vec<f32> {
    let mut list = vec![];
    for line in io::BufReader::new(File::open(filename).expect("failed to open file")).lines() {
        list.push(
            line.unwrap()
                .parse::<f32>()
                .expect("failed to parse float value"),
        );
    }
    list
}

fn mnist2(root: &str) -> Model2 {
    let mut m = Model2::default();

    let conv0_in = m.values.new_val(); // Input tensor [1, 1, 28, 28]
    let conv0_weight = m.values.new_val();
    let conv0_out = m.values.new_val();
    let _conv0 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![].into())
        .with_in(conv0_in)
        .with_in(conv0_weight)
        .with_out(conv0_out)
        .alloc(&mut m.nodes);

    let add0_const = m.values.new_val();
    let add0_out = m.values.new_val();
    let _add0 = Node2::new(Op::Add)
        .with_in(conv0_out)
        .with_in(add0_const)
        .with_out(add0_out)
        .alloc(&mut m.nodes);

    let relu0_out = m.values.new_val();
    let _relu0 = Node2::new(Op::ReLU)
        .with_in(add0_out)
        .with_out(relu0_out)
        .alloc(&mut m.nodes);

    let maxpool0_out = m.values.new_val();
    let _maxpool0 = Node2::new(Op::MaxPool)
        .with_attr(vec![2, 2].into())
        .with_attr(vec![2, 2].into())
        .with_in(relu0_out)
        .with_out(maxpool0_out)
        .alloc(&mut m.nodes);

    let conv1_weight = m.values.new_val();
    let conv1_out = m.values.new_val();
    let _conv1 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![2, 2].into())
        .with_in(maxpool0_out)
        .with_in(conv1_weight)
        .with_out(conv1_out)
        .alloc(&mut m.nodes);

    let add1_const = m.values.new_val();
    let add1_out = m.values.new_val();
    let _add1 = Node2::new(Op::Add)
        .with_in(conv1_out)
        .with_in(add1_const)
        .with_out(add1_out)
        .alloc(&mut m.nodes);

    let relu1_out = m.values.new_val();
    let _relu1 = Node2::new(Op::ReLU)
        .with_in(add1_out)
        .with_out(relu1_out)
        .alloc(&mut m.nodes);

    let maxpool1_out = m.values.new_val();
    let _maxpool1 = Node2::new(Op::MaxPool)
        .with_in(relu1_out)
        .with_out(maxpool1_out)
        .with_attr(vec![3, 3].into())
        .with_attr(vec![3, 3].into())
        .alloc(&mut m.nodes);

    let reshape0_const = m.values.new_val();
    let reshape0_out = m.values.new_val();
    let _reshape0 = Node2::new(Op::Reshape)
        .with_in(maxpool1_out)
        .with_in(reshape0_const)
        .with_out(reshape0_out)
        .alloc(&mut m.nodes);

    let reshape1_const0 = m.values.new_val();
    let reshape1_const1 = m.values.new_val();
    let reshape1_out = m.values.new_val();
    let _reshape1 = Node2::new(Op::Reshape)
        .with_in(reshape1_const0)
        .with_in(reshape1_const1)
        .with_out(reshape1_out)
        .alloc(&mut m.nodes);

    let matmul0_out = m.values.new_val();
    let _matmul0 = Node2::new(Op::MatMul)
        .with_in(reshape0_out)
        .with_in(reshape1_out)
        .with_out(matmul0_out)
        .alloc(&mut m.nodes);

    let add2_const = m.values.new_val();
    let add2_out = m.values.new_val();
    let _add2 = Node2::new(Op::Add)
        .with_in(matmul0_out)
        .with_in(add2_const)
        .with_out(add2_out)
        .alloc(&mut m.nodes);

    m.inputs.push(conv0_in);
    m.outputs.push(add2_out);

    m.inits.insert(
        add0_const,
        Tensor2::new(vec![8, 1, 1].into())
            .with_data(read_f32_list_from(Path::new(root).join("add1").to_str().unwrap()).into()),
    );
    m.inits.insert(
        add1_const,
        Tensor2::new(vec![16, 1, 1].into())
            .with_data(read_f32_list_from(Path::new(root).join("add2").to_str().unwrap()).into()),
    );
    m.inits.insert(
        add2_const,
        Tensor2::new(vec![1, 10].into())
            .with_data(read_f32_list_from(Path::new(root).join("add3").to_str().unwrap()).into()),
    );
    m.inits.insert(
        conv0_weight,
        Tensor2::new(vec![8, 1, 5, 5].into())
            .with_data(read_f32_list_from(Path::new(root).join("conv1").to_str().unwrap()).into()),
    );
    m.inits.insert(
        conv1_weight,
        Tensor2::new(vec![16, 8, 5, 5].into())
            .with_data(read_f32_list_from(Path::new(root).join("conv2").to_str().unwrap()).into()),
    );
    m.inits.insert(
        reshape0_const,
        Tensor2::new(vec![2].into()).with_data(vec![1, 256].into()),
    );
    m.inits.insert(
        reshape1_const0,
        Tensor2::new(vec![16, 4, 4, 10].into()).with_data(
            read_f32_list_from(Path::new(root).join("reshape1").to_str().unwrap()).into(),
        ),
    );
    m.inits.insert(
        reshape1_const1,
        Tensor2::new(vec![2].into()).with_data(vec![256, 10].into()),
    );

    m
}

fn mnist(root: &str) -> Model {
    let mut m = Model::new();
    let input = m.new(Node::Input(vec![1, 1, 28, 28].into()));
    let conv_weight = m.new(
        Tensor::new(vec![8, 1, 5, 5].into())
            .with_data(read_f32_list_from(Path::new(root).join("conv1").to_str().unwrap()).into())
            .into(),
    );
    let conv = m.new_conv2d(
        input,
        conv_weight,
        vec![5, 5].into(),
        vec![1, 1].into(),
        vec![2, 2].into(),
    );
    let add_input_b = m.new(
        Tensor::new(vec![8, 1, 1].into())
            .with_data(read_f32_list_from(Path::new(root).join("add1").to_str().unwrap()).into())
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
    let relu = m.new_relu(add);
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
            .with_data(read_f32_list_from(Path::new(root).join("conv2").to_str().unwrap()).into())
            .into(),
    );
    let conv2 = m.new_conv2d(
        max_pool,
        conv2_weight,
        vec![5, 5].into(),
        vec![1, 1].into(),
        vec![2, 2].into(),
    );
    let add2_input_b = m.new(
        Tensor::new(vec![16, 1, 1].into())
            .with_data(read_f32_list_from(Path::new(root).join("add2").to_str().unwrap()).into())
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
    let relu2 = m.new_relu(add2);
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
                read_f32_list_from(Path::new(root).join("reshape1").to_str().unwrap()).into(),
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
            .with_data(read_f32_list_from(Path::new(root).join("add3").to_str().unwrap()).into())
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
