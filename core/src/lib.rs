#![allow(clippy::excessive_precision)]

pub mod analysis;
pub mod dim;
pub mod symdim;
pub mod model;
pub mod node;
pub mod onnx;
pub mod op;
pub mod optimize;
pub mod tensor;
pub mod value;

#[cfg(not(target_arch = "wasm32"))]
use mimalloc::MiMalloc;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
