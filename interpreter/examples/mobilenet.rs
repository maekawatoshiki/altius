use altius_core::onnx::load_onnx;
// use altius_core::tensor::*;
// use altius_interpreter::Interpreter2;
// use rayon::prelude::*;
// use std::cmp::Ordering;
// use std::fs;
use std::path::Path;

fn main() {
    env_logger::init();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let _model = load_onnx(root.join("mobilenetv3.onnx")).unwrap();
}
