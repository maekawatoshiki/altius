#![feature(portable_simd)]

pub mod interpreter;

#[cfg(all(feature = "cblas", not(feature = "blis")))]
#[allow(unused)]
use blas_src;
#[cfg(all(feature = "cblas", feature = "blis"))]
#[allow(unused)]
use blis_src;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {}
