use std::ptr;

use altius_core::{model::Model, node::NodeId, tensor::Tensor, value::ValueId};
use opencl3::{
    command_queue::{enqueue_read_buffer, enqueue_write_buffer, CommandQueue},
    context::Context,
    device::Device,
    kernel::{ExecuteKernel, Kernel},
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
                enqueue_write_buffer(
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

        let mut events = vec![];

        for plan in &self.execution_plans {
            let node = &self.model.nodes[plan.node_id];
            let mut exec_kernel = ExecuteKernel::new(&plan.kernel);

            let mut size = 0;
            for &output in &node.outputs {
                let t = &self.values[&output].tensor;
                size = t.dims().total_elems();
                unsafe { exec_kernel.set_arg(&self.values[&output].buf) };
            }

            for &input in &node.inputs {
                unsafe { exec_kernel.set_arg(&self.values[&input].buf) };
            }

            events.push(
                unsafe {
                    exec_kernel
                        .set_global_work_size(size)
                        .enqueue_nd_range(&self.queue)
                }
                .unwrap(),
            );
        }

        let mut outputs = vec![];

        for out_id in &self.model.outputs {
            let val = &self.values[out_id];
            let out = Tensor::uninit_of_type(val.tensor.elem_ty(), val.tensor.dims().clone());

            let _event = unsafe {
                enqueue_read_buffer(
                    self.queue.get(),
                    val.buf,
                    CL_BLOCKING,
                    0,
                    val.tensor.elem_ty().size() * val.tensor.dims().total_elems(),
                    out.data_as_ptr() as *mut _,
                    0,
                    ptr::null(),
                )
            }
            .map_err(|e| {
                SessionError::Message(format!("Failed to read buffer: {}", e.to_string()).into())
            })?;

            outputs.push(out)
        }

        for event in events {
            let start_time = event.profiling_command_start().unwrap();
            let end_time = event.profiling_command_end().unwrap();
            let duration = end_time - start_time;
            println!(
                "kernel execution duration (ms): {}",
                duration as f64 / 1000. / 1000.0
            );
        }

        Ok(outputs)
    }
}
