use altius_core::onnx::load_onnx;
use altius_core::optimize::gelu_fusion::fuse_gelu;
use altius_core::tensor::Tensor;
use altius_session::interpreter::Interpreter;
use std::path::PathBuf;
use std::process::exit;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compile")]
pub struct Opt {
    #[structopt(parse(from_os_str))]
    pub onnx_path: PathBuf,

    #[structopt(long = "profile", help = "Enable profiling")]
    pub profile: bool,

    #[structopt(
        long = "threads",
        help = "The number of threads for computation",
        default_value = "1"
    )]
    pub threads: usize,
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();

    log::info!("load onnx: start ({:?})", opt.onnx_path);
    let start = Instant::now();
    let mut model = load_onnx(opt.onnx_path).unwrap();
    log::info!("load onnx: finished in {:?}", start.elapsed());

    fuse_gelu(&mut model);

    log::info!(
        "create session: start (profile={:?}, threads={})",
        opt.profile,
        opt.threads
    );
    let start = Instant::now();
    let sess = Interpreter::new(&model)
        .with_profiling(opt.profile)
        .with_intra_op_num_threads(opt.threads);
    log::info!("create session: finished in {:?}", start.elapsed());

    let mut inputs = vec![];
    for (i, &input_id) in model.inputs.iter().enumerate() {
        if model.inits.contains_key(&input_id) {
            continue;
        }

        let input = &model.values.inner()[input_id];
        let name = input.name.as_ref().map(String::as_str).unwrap_or("");
        let Some(shape) = input.shape.as_ref() else {
            log::info!("failed to feed input({i}, name={name}): unknown shape");
            exit(1);
        };

        log::info!(
            "feed input({i}, name={}, ty={:?}, shape={:?}): random input",
            name,
            shape.elem_ty,
            shape.dims
        );

        inputs.push((
            input_id,
            Tensor::rand_of_type(shape.elem_ty, shape.dims.clone()),
        ));
    }

    let outputs = match sess.run(inputs) {
        Ok(outputs) => outputs,
        Err(e) => {
            log::info!("inference failed: {:?}", e);
            exit(1);
        }
    };

    for (i, (output, output_id)) in outputs.iter().zip(model.outputs.iter()).enumerate() {
        let name = model.values.inner()[*output_id]
            .name
            .as_ref()
            .map(String::as_str)
            .unwrap_or("");
        log::info!("output({i}, name={}, shape={:?}): ...", name, output.dims());
    }
}
