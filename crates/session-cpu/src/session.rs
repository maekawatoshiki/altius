use altius_core::{
    model::Model,
    tensor::{Tensor, TypedFixedShape},
    value::ValueId,
};
use altius_session::SessionError;
use rustc_hash::FxHashMap;

use std::{path::PathBuf, time::Instant};

pub struct CPUSession {
    pub(super) model: Model,
    #[allow(dead_code)]
    pub(super) target_dir: PathBuf,
    pub(super) value_shapes: FxHashMap<ValueId, TypedFixedShape>,
    #[allow(dead_code)]
    pub(super) lib: libloading::Library,
    pub(super) trampoline: extern "C" fn(*const *const u8, *const *mut u8),
    pub(super) enable_profiling: bool,
    pub(super) profile_symbols: FxHashMap<String, *const f64>,
}

// TODO: Is this really safe?
unsafe impl Send for CPUSession {}

impl CPUSession {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError> {
        let mut outputs = self
            .model
            .graph
            .outputs
            .iter()
            .map(|id| {
                let shape = &self.value_shapes[id];
                Tensor::uninit_of_type(shape.elem_ty, shape.dims.clone())
            })
            .collect::<Vec<_>>();

        let start = Instant::now();

        {
            let mut inputs_ = Vec::with_capacity(inputs.len());
            let mut outputs_ = Vec::with_capacity(outputs.len());
            for tensor in inputs.iter() {
                inputs_.push(tensor.data_as_ptr());
            }
            for tensor in outputs.iter_mut() {
                outputs_.push(tensor.data_as_mut_ptr());
            }
            (self.trampoline)(inputs_.as_ptr(), outputs_.as_ptr());
        }

        if self.enable_profiling {
            let entire_duration = start.elapsed().as_secs_f32() * 1000.0;
            let mut durations = self
                .profile_symbols
                .iter()
                .map(|(name, &duration)| {
                    let duration = unsafe { *duration };
                    (name.as_str(), duration as f32 * 1000.0)
                })
                .collect::<Vec<_>>();
            let sum_durations = durations.iter().map(|(_, d)| d).sum::<f32>();
            durations.push(("All", entire_duration));
            durations.push(("All (Kernel)", sum_durations));
            durations.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
            let width = durations.iter().map(|(op, _)| op.len()).max().unwrap();
            for (op, duration) in durations {
                log::info!("| {op:width$}: {duration:.5} [ms]");
            }
        }

        Ok(outputs)
    }
}
