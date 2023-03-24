use altius_core::{
    analysis::shape::infer_shapes,
    model::Model,
    node::NodeId,
    op::Op,
    tensor::{Tensor, TypedShape},
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
        let model = self.model;
        let enable_profiling = self.enable_profiling;
        let intra_op_num_threads = self.intra_op_num_threads;

        let sorted_nodes = model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        infer_shapes(
            &model,
            &sorted_nodes,
            &mut inferred_shapes,
            &mut FxHashMap::default(),
        )?;

        let execution_plans = create_execution_plan(&model, &sorted_nodes);
        translate_into_c(&model, &execution_plans, &inferred_shapes)?;

        Ok(CPUSession {
            execution_plans: create_execution_plan(&model, &sorted_nodes),
            model,
            inferred_shapes,
            enable_profiling,
            values: ThreadLocal::new(),
            dummy_value: Tensor::zeros::<f32>(vec![0].into()),
            tctx: ThreadCtx::new_with_num_threads(intra_op_num_threads),
        })
    }
}

fn translate_into_c(
    _model: &Model,
    _execution_plans: &[NodeExecutionPlan],
    _inferred_shapes: &FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
) -> Result<(), SessionError> {
    todo!()
}
