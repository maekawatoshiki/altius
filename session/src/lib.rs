pub mod interpreter;

#[cfg(feature = "cblas")]
#[allow(unused)]
use blas_src;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {}
