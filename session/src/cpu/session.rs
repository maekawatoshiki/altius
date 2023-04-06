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
    #[allow(dead_code)]
    pub(super) target_dir: PathBuf,
    pub(super) value_shapes: FxHashMap<ValueId, TypedShape>,
    #[allow(dead_code)]
    pub(super) lib: libloading::Library,
    pub(super) entry: *const std::ffi::c_void,
    pub(super) enable_profiling: bool,
    pub(super) profile_symbols: FxHashMap<String, *const f64>,
}

// TODO: Is this really safe?
unsafe impl Send for CPUSession {}

impl CPUSession {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        assert!(matches!(inputs.len(), 1 | 2 | 3));
        assert_eq!(self.model.outputs.len(), 1);

        let mut outputs = self
            .model
            .outputs
            .iter()
            .map(|id| {
                let shape = &self.value_shapes[id];
                Tensor::uninit_of_type(shape.elem_ty, shape.dims.clone())
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        unsafe {
            // TODO: Use macro or something better.
            match (inputs.len(), outputs.len()) {
                (1, 1) => {
                    let entry = std::mem::transmute::<_, unsafe extern "C" fn(*const u8, *mut u8)>(
                        self.entry,
                    );
                    entry(inputs[0].1.data_as_ptr(), outputs[0].data_as_mut_ptr());
                }
                (2, 1) => {
                    let entry = std::mem::transmute::<
                        _,
                        unsafe extern "C" fn(*const u8, *const u8, *mut u8),
                    >(self.entry);
                    entry(
                        inputs[0].1.data_as_ptr(),
                        inputs[1].1.data_as_ptr(),
                        outputs[0].data_as_mut_ptr(),
                    );
                }
                (3, 1) => {
                    let entry = std::mem::transmute::<
                        _,
                        unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8),
                    >(self.entry);
                    entry(
                        inputs[0].1.data_as_ptr(),
                        inputs[1].1.data_as_ptr(),
                        inputs[2].1.data_as_ptr(),
                        outputs[0].data_as_mut_ptr(),
                    );
                }
                _ => unreachable!(),
            }
        };

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
