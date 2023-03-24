use std::{
    fs::{self, File},
    io::{BufWriter, Write as _},
    path::PathBuf,
};

use altius_core::{
    analysis::shape::infer_shapes,
    model::Model,
    node::NodeId,
    op::{Conv2d, Flatten, Gemm, HardSigmoid, MaxPool, Op},
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;

use crate::{create_execution_plan, NodeExecutionPlan, SessionError};

#[cfg(feature = "cuda")]
use super::session::SafeCudnnContext;
use super::{session::CPUSession, thread::ThreadCtx};
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;

pub struct CPUSessionBuilder {
    model: Model,
    intra_op_num_threads: usize,
    enable_profiling: bool,
}

impl CPUSessionBuilder {
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

    pub fn build(self) -> Result<CPUSession, SessionError> {
        let sorted_nodes = self.model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        let mut value_shapes = FxHashMap::default();
        infer_shapes(
            &self.model,
            &sorted_nodes,
            &mut inferred_shapes,
            &mut value_shapes,
        )?;

        let execution_plans = create_execution_plan(&self.model, &sorted_nodes);

        Translator::new(&self.model, &inferred_shapes, &value_shapes)?
            .translate_into_c(&execution_plans)?;

        Ok(CPUSession {
            execution_plans,
            model: self.model,
            inferred_shapes,
            enable_profiling: self.enable_profiling,
            values: ThreadLocal::new(),
            dummy_value: Tensor::zeros::<f32>(vec![0].into()),
            tctx: ThreadCtx::new_with_num_threads(self.intra_op_num_threads),
        })
    }
}

struct Translator<'a> {
    model: &'a Model,
    inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    created_calls: Vec<String>,
    created_file_paths: Vec<PathBuf>,
    tempdir: tempfile::TempDir,
    save_generated_files: bool,
}

impl<'a> Translator<'a> {
    fn new(
        model: &'a Model,
        inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
        value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            model,
            inferred_shapes,
            value_shapes,
            created_calls: Vec::new(),
            created_file_paths: Vec::new(),
            tempdir: tempfile::tempdir()?,
            save_generated_files: true,
        })
    }

    fn translate_into_c(
        &mut self,
        execution_plans: &[NodeExecutionPlan],
    ) -> Result<(), SessionError> {
        let main_file = self.create_file("main.c")?;
        let mut writer = BufWriter::new(main_file);

        for plan in execution_plans {
            self.translate_node(plan.node_id)?;
        }

        {
            let headers = format!(
                "#include <assert.h>
#include <blis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>"
            );
            writer.write_all(headers.as_bytes())?;
        }

        writer.flush()?;

        if self.save_generated_files {
            let dir = "/tmp/model";
            let _ = fs::remove_dir_all(dir); // Ignore errors if the directory does not exist.
            fs::create_dir(dir)?;
            for path in &self.created_file_paths {
                let name = path.file_name().unwrap();
                fs::copy(path, PathBuf::from(dir).join(name)).unwrap();
            }
        }

        Ok(())
    }

    fn translate_node(&mut self, node_id: NodeId) -> Result<(), SessionError> {
        let node = &self.model.nodes[node_id];
        let inputs = node
            .inputs
            .iter()
            .map(|value_id| &self.value_shapes[value_id])
            .collect::<Vec<_>>();
        let (op, outputs) = self.inferred_shapes.get(&node_id).cloned().map_or_else(
            || todo!("Why is this node output shape not inferred?"),
            |result| Ok::<(Op, Vec<TypedShape>), SessionError>(result),
        )?;

        let node_name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("{}_noname_{}", node.op.name(), node_id.index()));
        log::debug!("Translating node: {}", node_name);

        match op {
            Op::Conv2d(ref c) => self.translate_conv2d(c, &inputs, &outputs)?,
            Op::HardSigmoid(ref h) => self.translate_hard_sigmoid(h, &inputs, &outputs)?,
            Op::Add => self.translate_add(&inputs, &outputs)?,
            Op::Mul => self.translate_mul(&inputs, &outputs)?,
            Op::ReLU => self.translate_relu(&inputs, &outputs)?,
            Op::GlobalAveragePool => self.translate_gavg_pool(&inputs, &outputs)?,
            Op::MaxPool(ref m) => self.translate_max_pool(m, &inputs, &outputs)?,
            Op::Flatten(ref f) => self.translate_flatten(f, &inputs, &outputs)?,
            Op::Gemm(ref g) => self.translate_gemm(g, &inputs, &outputs)?,
            _ => todo!("Translation not implemented for {:?}", op),
        };

        Ok(())
    }

    fn translate_conv2d(
        &mut self,
        _conv2d: &Conv2d,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_hard_sigmoid(
        &mut self,
        _hs: &HardSigmoid,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_add(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_mul(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_relu(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_gavg_pool(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_max_pool(
        &mut self,
        _max_pool: &MaxPool,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_flatten(
        &mut self,
        _flatten: &Flatten,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_gemm(
        &mut self,
        _gemm: &Gemm,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn create_file(&mut self, name: &str) -> Result<File, SessionError> {
        let path = self.tempdir.path().join(name);
        let file = fs::File::create(&path)?;
        self.created_file_paths.push(path);
        Ok(file)
    }
}
