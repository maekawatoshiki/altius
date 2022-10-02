pub mod dim;
pub mod model;
pub mod node;
pub mod onnx;
pub mod optimize;
pub mod tensor;
pub mod value;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
