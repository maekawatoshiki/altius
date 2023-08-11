use altius_core::optimize::gelu_fusion::fuse_gelu;
use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_session_interpreter::InterpreterSessionBuilder;
use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compile")]
pub struct Opt {
    #[structopt(long = "profile", help = "Enable profiling")]
    pub profile: bool,

    #[structopt(long = "iters", help = "The number of iterations", default_value = "1")]
    pub iters: usize,
}

fn main() {
    env_logger::init();
    color_backtrace::install();

    let opt = Opt::from_args();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let mut model = load_onnx(root.join("vit_b_16.onnx")).unwrap();
    fuse_gelu(&mut model);

    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 384, 384, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 384, 384), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(vec![1, 3, 384, 384].into(), image.into_raw_vec());

    let i = InterpreterSessionBuilder::new(model)
        .with_profiling_enabled(opt.profile)
        .with_intra_op_num_threads(16)
        .build()
        .unwrap();
    for _ in 0..opt.iters {
        let out = i.run(vec![input.clone()]).expect("Inference failed");
        let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
        out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let classes = fs::read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
        let classes = classes.split('\n').collect::<Vec<_>>();
        println!("inferred: {}", classes[out[0].0]);
        println!("top5: {:?}", &out[..5]);
    }
}
