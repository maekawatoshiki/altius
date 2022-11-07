#![feature(portable_simd)]

pub mod interpreter;
pub mod opencl;
pub mod wgpu;

use std::fmt::Display;

#[cfg(all(feature = "cblas", not(feature = "blis")))]
#[allow(unused)]
use blas_src;
#[cfg(all(feature = "cblas", feature = "blis"))]
#[allow(unused)]
use blis_src;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {
    Message(String),
}

impl Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(msg) => writeln!(f, "{msg}"),
        }
    }
}
