#![feature(portable_simd)]
#![allow(clippy::excessive_precision)]

mod builder;
mod conv2d;
mod fast_math;
mod gemm;
mod session;
mod thread;

pub use builder::InterpreterSessionBuilder;
pub use session::InterpreterSession;
