use std::{cmp::Ordering, fs::read_to_string, path::Path, time::Instant};

use ndarray::CowArray;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compile")]
pub struct Opt {
    #[structopt(long = "profile", help = "Enable profiling")]
    pub profile: bool,

    #[structopt(long = "iters", help = "The number of iterations", default_value = "1")]
    pub iters: usize,

    #[structopt(
        long = "threads",
        help = "The number of computation threads",
        default_value = "1"
    )]
    pub threads: usize,

    #[structopt(long = "ort", help = "Use ONNX Runtime")]
    pub ort: bool,
}

fn main() {
    use altius_core::optimize::elemwise_fusion::fuse_elemwise_ops;
    use altius_core::optimize::gelu_fusion::fuse_gelu;
    use altius_core::optimize::layer_norm_fusion::fuse_layer_norm;
    use altius_core::{onnx::load_onnx, tensor::Tensor};
    use altius_session_clang::CPUSessionBuilder;
    use std::fs;

    env_logger::init();
    color_backtrace::install();

    let opt = Opt::from_args();

    if opt.ort {
        return run_on_ort(&opt);
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut model = load_onnx(root.join("deit.onnx"))
        .expect("Failed to load model. Have you run altius_py/deit.py?");
    fuse_layer_norm(&mut model);
    fuse_gelu(&mut model);
    fuse_elemwise_ops(&mut model).unwrap();

    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.5, 0.5, 0.5][c];
        let std = [0.5, 0.5, 0.5][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = Tensor::new(vec![1, 3, 224, 224].into(), image.into_raw_vec());

    let i = CPUSessionBuilder::new(model)
        .with_profiling_enabled(opt.profile)
        .with_intra_op_num_threads(opt.threads)
        .build()
        .unwrap();
    let classes = fs::read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
    let classes = classes.split('\n').collect::<Vec<_>>();
    for _ in 0..opt.iters {
        let out = i.run(vec![input.clone()]).expect("Inference failed");
        let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
        out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        println!("inference result: {}", classes[out[0].0]);
        println!("top5: {:?}", &out[..5]);
    }
}

fn run_on_ort(opt: &Opt) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let env = Environment::builder()
        .with_name("altius")
        .build()
        .unwrap()
        .into_arc();
    let session = SessionBuilder::new(&env)
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(opt.threads as i16)
        .unwrap()
        .with_model_from_file(root.join("deit.onnx"))
        .unwrap();

    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.5, 0.5, 0.5][c];
        let std = [0.5, 0.5, 0.5][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = CowArray::from(image)
        .into_shape((1, 3, 224, 224))
        .unwrap()
        .into_dimensionality()
        .unwrap();

    let classes = read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
    for _ in 0..opt.iters {
        let input = Value::from_array(session.allocator(), &input).unwrap();
        let now = Instant::now();
        let output = &session.run(vec![input]).unwrap()[0];
        let elapsed = now.elapsed();
        let output = output.try_extract::<f32>().unwrap();
        let output = output.view();
        let mut output = output.iter().enumerate().collect::<Vec<_>>();
        output.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        println!(
            "inferred: {} (in {elapsed:?})",
            classes.split('\n').collect::<Vec<_>>()[output[0].0],
        );
    }
}
