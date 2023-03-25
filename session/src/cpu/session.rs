use crate::SessionError;
use altius_core::{
    model::Model,
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
use rustc_hash::FxHashMap;

use std::{path::PathBuf, time::Instant};

pub struct CPUSession {
    pub(super) model: Model,
    pub(super) target_dir: PathBuf,
    pub(super) value_shapes: FxHashMap<ValueId, TypedShape>,
}

impl CPUSession {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        assert_eq!(inputs.len(), 1);

        let lib = unsafe { libloading::Library::new(self.target_dir.join("model.so")) }?;
        let func: libloading::Symbol<unsafe extern "C" fn(*const f32, *mut f32) -> u32> =
            unsafe { lib.get(b"model_entry")? };
        let mut outputs = self
            .model
            .outputs
            .iter()
            .map(|id| {
                let shape = &self.value_shapes[id];
                Tensor::uninit_of_type(shape.elem_ty, shape.dims.clone())
            })
            .collect::<Vec<_>>();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = unsafe {
                func(
                    inputs[0].1.data().as_ptr(),
                    outputs[0].data_mut().as_mut_ptr(),
                )
            };
            log::debug!("elapsed: {:?}", start.elapsed());
        }

        Ok(outputs)
    }
}
