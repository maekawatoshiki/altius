pub mod interpreter;

#[allow(unused)]
use blas_src;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {}
