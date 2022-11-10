use std::{cmp::Ordering, io::Cursor};

use altius_core::{onnx::load_onnx_from_buffer, tensor::Tensor};
use altius_session::interpreter::InterpreterSessionBuilder;
use image::io::Reader;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn load_and_run(onnx: &[u8], img: &[u8]) -> String {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let model = load_onnx_from_buffer(onnx).expect("failed to load onnx");
    let sess = InterpreterSessionBuilder::new(&model).build();
    let input_value = model
        .lookup_named_value("input")
        .expect("failed to lookup input value");

    let image = Reader::new(Cursor::new(img))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(
        vec![1, 3, 224, 224].into(),
        image.clone().into_raw_vec().into_iter().collect::<Vec<_>>(),
    );

    let results = sess
        .run(vec![(input_value, input)])
        .expect("failed to run inference");

    let mut out = results[0]
        .data::<f32>()
        .iter()
        .enumerate()
        .collect::<Vec<_>>();
    out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    let classes = include_str!("../../models/imagenet_classes.txt");
    let classes = classes.split('\n').collect::<Vec<_>>();
    let mut result_str = "".to_string();
    for i in 0..5 {
        result_str.push_str(&format!("top{}: {}<br>", i + 1, classes[out[i].0]));
    }

    result_str
}
