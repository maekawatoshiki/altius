use std::{fs::File, io::Write};

use altius_core::node::NodeId;
use rustc_hash::FxHashMap;
use tempfile::TempDir;

pub struct CodegenCtx {
    tempdir: TempDir,
    node_to_kernel: FxHashMap<NodeId, libloading::os::unix::Symbol<Kernel>>,
}

pub type Kernel = unsafe extern "C" fn();

impl CodegenCtx {
    pub fn new() -> Option<Self> {
        Some(Self {
            tempdir: tempfile::tempdir().ok()?,
            node_to_kernel: FxHashMap::default(),
        })
    }

    pub fn compile(&mut self, node_id: NodeId, kernel: String) {
        let mut file = File::create(self.tempdir.path().join("program.c")).unwrap();
        file.write_all(kernel.as_bytes()).unwrap();
        file.flush().unwrap();

        std::process::Command::new("gcc")
            .arg("-shared")
            .arg("-fPIC")
            .arg("-O3")
            .arg("-march=native")
            .arg("-fopenmp")
            .arg(self.tempdir.path().join("program.c"))
            .arg("-o")
            .arg(self.tempdir.path().join("program.so"))
            .output()
            .unwrap();

        unsafe {
            let lib = libloading::Library::new(self.tempdir.path().join("program.so")).unwrap();
            let kernel: libloading::Symbol<Kernel> = lib.get(b"kernel").unwrap();
            self.node_to_kernel.insert(node_id, kernel.into_raw());
        }
    }
}
