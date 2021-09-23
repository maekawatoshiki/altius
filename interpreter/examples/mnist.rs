use altius_core::tensor::*;
use altius_interpreter::Interpreter;
use altius_onnx::load_onnx_model;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::env::args;
use std::fs;
use std::path::Path;

fn main() {
    let model_root = &args().collect::<Vec<String>>()[1];
    let mnist = load_onnx_model(Path::new(model_root).join("mnist-8.onnx"));

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
    let validation_count = 1000;
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
