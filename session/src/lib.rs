#![feature(portable_simd)]
#![allow(clippy::excessive_precision)]

#[cfg(feature = "opencl")]
pub mod opencl;
pub mod plan;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu;

use std::borrow::Cow;

use altius_core::{analysis::shape::ShapeError, tensor::Tensor};
#[cfg(all(feature = "cblas", target_os = "macos"))]
#[allow(unused)]
#[allow(clippy::single_component_path_imports)]
use blas_src; // For accelerate, this is necessary to link the library.
#[cfg(all(feature = "cblas", target_os = "linux"))]
#[allow(unused)]
#[allow(clippy::single_component_path_imports)]
use blis_src;
use cranelift_module::ModuleError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    /// Errors arised from shape inference.
    #[error("Shape: {0}")]
    Shape(#[from] ShapeError),

    #[error("Io: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "cpu-backend")]
    #[error("Libloading: {0}")]
    Libloading(#[from] libloading::Error),

    #[cfg(feature = "cpu-backend")]
    #[error("Cranelift: {0}")]
    Cranelift(#[from] ModuleError),

    /// General error messages (including TODOs).
    #[error("Something went wrong: {0}")]
    Message(Cow<'static, str>),
}

pub trait Session {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError>;
}
