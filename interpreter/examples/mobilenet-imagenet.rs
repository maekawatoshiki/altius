use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_interpreter::Interpreter;
use indicatif::ProgressBar;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::cmp::Ordering;
// use std::fs;
use std::path::Path;
use structopt::StructOpt;
use walkdir::WalkDir;

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

    // let mut inputs = vec![];
    let mut input_files = vec![];
    let max_images = 500;
    for (i, p) in WalkDir::new("/home/uint/work/aoha/dataset_imagenet/validation")
        .into_iter()
        .enumerate()
    {
        if i >= max_images {
            break;
        }
        let p = p.unwrap();
        if p.path().is_dir() {
            continue;
        }
        input_files.push(p);
    }

    let bar = ProgressBar::new(max_images as u64);
    let inputs: Vec<_> = input_files
        .par_iter()
        .map(|p| {
            let class = (&p
                .path()
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()[6..9])
                .parse::<usize>()
                .unwrap();
            let image = image::open(p.path()).unwrap().to_rgb8();
            let resized =
                image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
            let image = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            });
            let input = Tensor::new(vec![1, 3, 224, 224].into(), image.into_raw_vec());
            bar.inc(1);
            (class, input)
            // inputs.push(input);
        })
        .collect();

    // let classes = fs::read_to_string(Path::new(&root).join("imagenet_classes.txt")).unwrap();
    // let classes = classes.split("\n").collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let i = Interpreter::new(&model).with_profiling(opt.profile);
    let bar = ProgressBar::new(max_images as u64);
    let (acc1, acc5): (i32, i32) = inputs
        .into_par_iter()
        .map(|(class, input)| {
            #[cfg(feature = "cuda")]
            Interpreter::new(&model).run(vec![(input_value, input.clone())]); // First run is slow so
                                                                              // ignore it.
            let out = i.run(vec![(input_value, input)]).unwrap();
            let mut out = out[0].data::<f32>().iter().enumerate().collect::<Vec<_>>();
            bar.inc(1);
            out.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            let idxes: Vec<_> = out[0..10].iter().map(|(x, _)| *x).collect();
            (
                idxes[0..1].contains(&class) as i32,
                idxes[0..5].contains(&class) as i32,
            )
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let elapsed = start.elapsed();
    println!(
        "acc@1 {}, acc@5 {}, elapsed {elapsed:?}",
        acc1 as f32 / max_images as f32,
        acc5 as f32 / max_images as f32
    );
}
