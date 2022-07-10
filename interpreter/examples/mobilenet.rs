use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_interpreter::Interpreter;
use std::cmp::Ordering;
use std::fs;
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

    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(vec![1, 3, 224, 224].into(), image.into_raw_vec());

    let mut i = Interpreter::new(&model).with_profiling(opt.profile);
    #[cfg(feature = "cuda")]
    Interpreter::new(&model).run(vec![(input_value, input.clone())]); // First run is slow so
                                                                      // ignore it.
    let out = i.run(vec![(input_value, input)]);
    let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
    out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    let classes = fs::read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
    let classes = classes.split("\n").collect::<Vec<_>>();
    println!("inferred: {}", classes[out[0].0]);
    println!("top5: {:?}", &out[..5]);
}
