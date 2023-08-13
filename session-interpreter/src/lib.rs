#![feature(portable_simd)]
#![allow(clippy::excessive_precision)]

#[cfg(all(feature = "cblas", target_os = "macos"))]
#[allow(unused)]
#[allow(clippy::single_component_path_imports)]
use blas_src; // For accelerate, this is necessary to link the library.
#[cfg(all(feature = "cblas", target_os = "linux"))]
#[allow(unused)]
#[allow(clippy::single_component_path_imports)]
use blis_src;

mod builder;
mod conv2d;
mod fast_math;
mod gemm;
mod session;
mod thread;

pub use builder::InterpreterSessionBuilder;
pub use session::InterpreterSession;
