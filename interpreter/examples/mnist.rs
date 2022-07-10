use altius_core::onnx::load_onnx;
use altius_core::tensor::*;
use altius_interpreter::Interpreter;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    env_logger::init();
    let mnist_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let mnist = load_onnx(mnist_root.join("mnist-8.onnx")).unwrap();

    let mut inputs = vec![];
    for line in fs::read_to_string(Path::new(&mnist_root).join("MNIST_test.txt"))
        .unwrap()
        .split("\n")
    {
        if line.is_empty() {
            continue;
        }
        let nums: Vec<&str> = line.split(",").collect();
        let expected: i32 = nums[0].parse().unwrap();
        let pixels: Tensor = Tensor::new(
            vec![1, 1, 28, 28].into(),
            nums[1..]
                .iter()
                .map(|s| s.parse::<f32>().unwrap() / 255.0)
                .collect::<Vec<_>>()
                .into(),
        );
        inputs.push((expected, pixels));
    }

    let mut correct: i32 = 0;
    let validation_count = 10000;
    let repeat = 1;

    let start = Instant::now();

    let input_value = mnist.lookup_named_value("Input3").unwrap();

    for _ in 0..repeat {
        correct = inputs
            .par_iter()
            .take(validation_count)
            .map(|(expected, input)| {
                let mut i = Interpreter::new(&mnist);
                let v = i.run(vec![(input_value, input.clone())]);
                let inferred = v[0]
                    .data::<f32>()
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
