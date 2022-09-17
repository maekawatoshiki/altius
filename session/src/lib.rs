pub mod interpreter;

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {}
