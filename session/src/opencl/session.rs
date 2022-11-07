use std::ptr;

use altius_core::{model::Model, node::NodeId, tensor::Tensor, value::ValueId};
use opencl3::{
    command_queue::{enqueue_read_buffer, CommandQueue},
    context::Context,
    device::Device,
    kernel::Kernel,
    types::CL_BLOCKING,
};
use rustc_hash::FxHashMap;

use crate::SessionError;

use super::tensor::OpenclTensor;

pub struct OpenclSession<'a> {
    /// Model
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) model: &'a Model,

    /// OpenCL device
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) device: Device,

    /// OpenCL context
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) context: Context,

    /// OpenCL queue
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) queue: CommandQueue,

    /// Mapping from `ValueId` to `OpenclTensor`
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) values: FxHashMap<ValueId, OpenclTensor>,

    #[allow(dead_code)] // TODO: Remove later.
    pub(super) execution_plans: Vec<ExecutionPlan>,
}

pub(super) struct ExecutionPlan {
    #[allow(dead_code)] // TODO: Remove later.
    pub(super) node_id: NodeId,

    #[allow(dead_code)] // TODO: Remove later
    pub(super) kernel: Kernel,
}

impl<'a> OpenclSession<'a> {
    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        for (input_id, input) in inputs {
            let val = &self.values[&input_id];

            let _event = unsafe {
                enqueue_read_buffer(
                    self.queue.get(),
                    val.buf,
                    CL_BLOCKING,
                    0,
                    val.tensor.elem_ty().size() * val.tensor.dims().total_elems(),
                    input.data_as_ptr() as *mut _,
                    0,
                    ptr::null(),
                )
                .unwrap()
            };
        }

        Ok(vec![])
        // todo!()
    }
}
