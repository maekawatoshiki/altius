use altius_core::{
    flops::compute_flops,
    model::Model,
    tensor::{Tensor, TypedFixedShape},
    value::ValueId,
};
use altius_session::SessionError;
use gecko_profile::{MarkerTiming, ProfileBuilder, ThreadBuilder, TracingMarker};
use rustc_hash::FxHashMap;

use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

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
    pub(super) profile: Profile,
}

#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub(super) events: Vec<(String, *const f64, *const f64)>,
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
        let start_sys = SystemTime::now();

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
            durations.push(("All (Kernel)", sum_durations));
            durations.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
            let width = durations.iter().map(|(op, _)| op.len()).max().unwrap();
            for (op, duration) in durations {
                log::info!("{op:width$}: {duration:.5} ms");
            }
            if let Ok(flops) = compute_flops(&self.model) {
                log::info!(
                    "[ {:.5} ms, {:.5} GFLOPS ]",
                    entire_duration,
                    flops as f32 / (entire_duration / 1000.0) / 1_000_000_000.0
                );
            }

            let mut s = ProfileBuilder::new(start, start_sys, "session", 0, Duration::from_secs(0));
            let mut t = ThreadBuilder::new(0, 0, start, true, false);
            let mut global_start = None;
            let mut last_end: Option<Duration> = None;
            for (i, &(ref name, s, e)) in self.profile.events.iter().enumerate() {
                let mut s = unsafe { *s };
                if global_start.is_none() {
                    global_start = Some(Duration::from_secs_f64(s));
                }
                let mut e = unsafe { *e };
                if s == 0.0 {
                    s = last_end.unwrap().as_secs_f64();
                }
                if e == 0.0 {
                    e = last_end.unwrap().as_secs_f64();
                }
                t.add_marker(
                    format!("{i:04}{name}").as_str(),
                    TracingMarker(),
                    MarkerTiming::Interval(
                        start + (Duration::from_secs_f64(s) - global_start.unwrap()),
                        start + (Duration::from_secs_f64(e) - global_start.unwrap()),
                    ),
                );
                last_end = Some(Duration::from_secs_f64(e));
            }
            // t.add_marker(
            //     "node2",
            //     TracingMarker(),
            //     MarkerTiming::Interval(Instant::now(), Instant::now()),
            // );
            // t.add_marker(
            //     "node3",
            //     TracingMarker(),
            //     MarkerTiming::Interval(Instant::now(), Instant::now()),
            // );
            s.add_thread(t);
            let s = s.to_serializable();
            File::create("profile.json")?
                .write_all(serde_json::to_string(&s).unwrap().as_bytes())?;
        }

        Ok(outputs)
    }
}
