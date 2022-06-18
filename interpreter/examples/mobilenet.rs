use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_interpreter::Interpreter2;
use std::cmp::Ordering;
use std::path::Path;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compile")]
pub struct Opt {
    #[structopt(long = "profile", help = "Enable profiling")]
    pub profile: bool,
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let model = load_onnx(root.join("mobilenetv3.onnx")).unwrap();
    let input_value = model.lookup_named_value("input").unwrap();

    let image = image::open(root.join("grace_hopper.jpg"))
        .unwrap()
        .to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(vec![1, 3, 224, 224].into()).with_data(image.into_raw_vec().into());

    let mut i = Interpreter2::new(&model).with_profiling(opt.profile);
    let out = i.run(vec![(input_value, input)]);
    let mut out = out
        .data()
        .as_f32()
        .unwrap()
        .iter()
        .enumerate()
        .collect::<Vec<_>>();
    out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    println!("top5: {:?}", &out[..5]);
}
