use altius_core::{onnx::load_onnx, tensor::Tensor};
// use altius_core::tensor::*;
use altius_interpreter::Interpreter2;
// use rayon::prelude::*;
// use std::cmp::Ordering;
// use std::fs;
use std::path::Path;

fn main() {
    env_logger::init();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let model = load_onnx(root.join("mobilenetv3.onnx")).unwrap();
    let input_value = model.lookup_named_value("input").unwrap();
    let input = Tensor::new(vec![1, 3, 224, 224].into());
    let mut i = Interpreter2::new(&model);
    let _v = i.run(vec![(input_value, input.clone())]);
}
