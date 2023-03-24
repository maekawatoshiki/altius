use crate::SessionError;
use altius_core::{model::Model, tensor::Tensor, value::ValueId};

use std::path::PathBuf;

pub struct CPUSession {
    pub(super) model: Model,
    pub(super) target_dir: PathBuf,
}

impl CPUSession {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        let lib = unsafe { libloading::Library::new(self.target_dir.join("model.so")) }?;
        let func: libloading::Symbol<unsafe extern "C" fn() -> u32> = unsafe { lib.get(b"main")? };
        let ret = unsafe { func() };
        // log::debug!("ret = {ret}");
        Ok(vec![])
        // Ok(self
        //     .model
        //     .outputs
        //     .iter()
        //     .map(|id| values.remove(id).unwrap())
        //     .collect())
    }
}
