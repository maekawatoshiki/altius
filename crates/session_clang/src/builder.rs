use altius_core::{analysis::shape::infer_shapes, model::Model};
use rustc_hash::FxHashMap;

use altius_session::SessionError;

use super::{session::ClangSession, translator::Translator};

pub struct ClangSessionBuilder {
    model: Model,
    intra_op_num_threads: usize,
    enable_profiling: bool,
}

impl ClangSessionBuilder {
    pub const fn new(model: Model) -> Self {
        Self {
            model,
            intra_op_num_threads: 1,
            enable_profiling: false,
        }
    }

    pub const fn with_intra_op_num_threads(mut self, intra_op_num_threads: usize) -> Self {
        self.intra_op_num_threads = intra_op_num_threads;
        self
    }

    pub const fn with_profiling_enabled(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    pub fn build(self) -> Result<ClangSession, SessionError> {
        let mut inferred_shapes = FxHashMap::default();
        let mut value_shapes = FxHashMap::default();
        infer_shapes(&self.model, &mut inferred_shapes, &mut value_shapes)?;

        let mut profile_symbols = FxHashMap::default();
        let product = Translator::new(&self.model, &inferred_shapes, &value_shapes)?
            .with_profiling_enabled(self.enable_profiling)
            .with_intra_op_num_threads(self.intra_op_num_threads)
            .compile()?;

        #[cfg(target_os = "linux")]
        let lib: libloading::Library = unsafe {
            // Load library with `RTLD_NOW | RTLD_NODELETE` to fix a SIGSEGV
            libloading::os::unix::Library::open(
                Some(product.target_dir.join("model.so")),
                0x2 | 0x1000,
            )?
            .into()
        };
        #[cfg(not(target_os = "linux"))]
        let lib = unsafe { libloading::Library::new(product.target_dir.join("model.so")) }?;
        {
            let initializer: libloading::Symbol<unsafe extern "C" fn()> =
                unsafe { lib.get(b"initialize")? };
            unsafe { initializer() };
        }
        let trampoline: libloading::Symbol<extern "C" fn(*const *const u8, *const *mut u8)> =
            unsafe { lib.get(b"trampoline")? };
        let trampoline = *trampoline;

        for (&val_id, tensor) in &self.model.graph.inits {
            let name = product.value_name(val_id);
            let entry: libloading::Symbol<*const *const u8> = unsafe { lib.get(name.as_bytes())? };
            unsafe { *entry.cast_mut() = tensor.data_as_ptr() };
        }

        if self.enable_profiling {
            for name in product.used_op_names {
                let symbol: libloading::Symbol<*const f64> =
                    unsafe { lib.get(format!("elapsed_{name}").as_bytes())? };
                profile_symbols.insert(name, unsafe { *symbol.into_raw() });
            }
        }

        Ok(ClangSession {
            target_dir: product.target_dir,
            model: self.model,
            lib,
            value_shapes,
            trampoline,
            enable_profiling: self.enable_profiling,
            profile_symbols,
        })
    }
}
