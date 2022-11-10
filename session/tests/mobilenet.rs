use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_session::interpreter::InterpreterSessionBuilder;
use std::cmp::Ordering;
use std::path::Path;

#[test]
fn mobilenet() {
    env_logger::init();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let mut model = load_onnx(root.join("mobilenetv3.onnx")).unwrap();
    let input_value = model.lookup_named_value("input").unwrap();

    // Change input batch size from 1 to 4.
    model.values.inner_mut()[input_value]
        .shape
        .as_mut()
        .unwrap()
        .dims
        .as_mut_slice()[0] = 4;

    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(
        vec![4, 3, 224, 224].into(),
        image
            .clone()
            .into_raw_vec()
            .into_iter()
            .chain(image.clone().into_iter())
            .chain(image.clone().into_iter())
            .chain(image.into_iter())
            .collect::<Vec<_>>(),
    );

    let i = InterpreterSessionBuilder::new()
        .with_model(&model)
        .with_profiling_enabled(true)
        .build()
        .expect("Failed to build interpreter session");
    #[cfg(feature = "cuda")]
    {
        use altius_session::interpreter::InterpreterSession;
        // First run is slow so ignore it.
        InterpreterSession::new(&model).run(vec![(input_value, input.clone())]);
    }
    let out = i.run(vec![(input_value, input)]).expect("Inference failed");
    let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
    out[0..1000].sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    out[1000..2000].sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    out[2000..3000].sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    out[3000..4000].sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    assert!(out[0].0 == 285 && out[1000].0 == 1285 && out[2000].0 == 2285 && out[3000].0 == 3285);
}
