use ndarray::CowArray;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compile")]
struct Opt {
    #[structopt(long = "profile", help = "Enable profiling")]
    profile: bool,

    #[structopt(long = "iters", help = "The number of iterations", default_value = "1")]
    iters: usize,

    #[structopt(
        long = "threads",
        help = "The number of computation threads",
        default_value = "1"
    )]
    threads: usize,

    #[structopt(long = "ort", help = "Use ONNX Runtime")]
    ort: bool,
}

fn main() {
    use altius_core::{onnx::load_onnx, tensor::Tensor};
    use altius_session_clang::ClangSessionBuilder;
    use std::cmp::Ordering;
    use std::fs;
    use std::path::Path;

    env_logger::init();
    color_backtrace::install();

    let opt = Opt::from_args();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");

    let classes = fs::read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
    let classes = classes.split('\n').collect::<Vec<_>>();
    let image = image::open(root.join("cat.png")).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });
    let input = CowArray::from(image.as_slice().unwrap())
        .into_shape((1, 3, 224, 224))
        .unwrap()
        .into_dimensionality()
        .unwrap();

    if opt.ort {
        let env = Environment::builder()
            .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])
            .build()
            .unwrap()
            .into_arc();
        let sess = SessionBuilder::new(&env)
            .unwrap()
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(8)
            .unwrap()
            .with_model_from_file(root.join("mobilenetv3.onnx"))
            .unwrap();
        for _ in 0..opt.iters {
            let x = Value::from_array(sess.allocator(), &input).unwrap();
            use std::time::Instant;
            let start = Instant::now();
            let out = &sess.run(vec![x]).unwrap()[0];
            log::info!("ort: {:?}", start.elapsed());
            let out = out.try_extract::<f32>().unwrap();
            let out = out.view();
            let out = out.as_slice().unwrap();
            let mut out = out.iter().enumerate().collect::<Vec<_>>();
            out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

            println!("prediction: {}", classes[out[0].0]);
            println!("top5: {:?}", &out[..5]);
        }
    } else {
        let model = load_onnx(root.join("mobilenetv3.onnx")).unwrap();
        let session = ClangSessionBuilder::new(model)
            .with_profiling_enabled(opt.profile)
            .with_intra_op_num_threads(opt.threads)
            .build()
            .unwrap();
        for _ in 0..opt.iters {
            let out = session
                .run(vec![Tensor::from(&input)])
                .expect("Inference failed");
            let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
            out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

            println!("prediction: {}", classes[out[0].0]);
            println!("top5: {:?}", &out[..5]);
        }
    }
}
