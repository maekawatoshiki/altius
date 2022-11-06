use altius_core::onnx::load_onnx;
use altius_core::optimize::gelu_fusion::fuse_gelu;
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
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();

    log::info!("load onnx: path = {:?}", opt.onnx_path);
    let start = Instant::now();
    let mut model = load_onnx(opt.onnx_path).unwrap();
    log::info!("load onnx: finished in {:?}", start.elapsed());

    fuse_gelu(&mut model);

    log::info!("todo!")
}
