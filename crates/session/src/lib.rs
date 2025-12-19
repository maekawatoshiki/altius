#![allow(clippy::excessive_precision)]

pub mod plan;

use std::borrow::Cow;

use altius_core::{analysis::shape::ShapeError, tensor::Tensor};
#[cfg(not(target_arch = "wasm32"))]
use cranelift_module::ModuleError;
#[cfg(not(target_arch = "wasm32"))]
use libloading;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    /// Errors arised from shape inference.
    #[error("Shape: {0}")]
    Shape(#[from] ShapeError),

    #[error("Io: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Libloading: {0}")]
    Libloading(#[from] libloading::Error),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Cranelift: {0}")]
    Cranelift(#[from] Box<ModuleError>),

    /// General error messages (including TODOs).
    #[error("Something went wrong: {0}")]
    Message(Cow<'static, str>),
}

#[cfg(not(target_arch = "wasm32"))]
impl From<ModuleError> for SessionError {
    fn from(err: ModuleError) -> Self {
        SessionError::Cranelift(Box::new(err))
    }
}

pub trait Session {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError>;
}
