use altius_core::onnx::load_onnx;
use altius_core::optimize::gelu_fusion::fuse_gelu;
use altius_session::interpreter::Interpreter;
use std::path::PathBuf;
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

    log::info!("todo! actual inference execution");
    // sess.run(inputs)
}
