#![feature(portable_simd)]

pub mod interpreter;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu;

use std::{borrow::Cow, fmt::Display};

#[cfg(all(feature = "cblas", not(feature = "blis")))]
#[allow(unused)]
use blas_src;
#[cfg(all(feature = "cblas", feature = "blis"))]
#[allow(unused)]
use blis_src;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {
    Message(Cow<'static, str>),
}

impl Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(msg) => writeln!(f, "{msg}"),
        }
    }
}
