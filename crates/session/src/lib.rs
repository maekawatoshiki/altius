#![allow(clippy::excessive_precision)]

pub mod plan;

use std::borrow::Cow;

use altius_core::{analysis::shape::ShapeError, tensor::Tensor};
use cranelift_module::ModuleError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    /// Errors arised from shape inference.
    #[error("Shape: {0}")]
    Shape(#[from] ShapeError),

    #[error("Io: {0}")]
    Io(#[from] std::io::Error),

    #[error("Libloading: {0}")]
    Libloading(#[from] libloading::Error),

    #[error("Cranelift: {0}")]
    Cranelift(#[from] ModuleError),

    /// General error messages (including TODOs).
    #[error("Something went wrong: {0}")]
    Message(Cow<'static, str>),
}

pub trait Session {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError>;
}
