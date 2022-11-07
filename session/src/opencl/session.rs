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
            let mut inputs = vec![];
            let mut outputs = vec![];
            for &input in &node.inputs {
                inputs.push(self.values[&input].buf);
            }
            let mut size = 0;
            assert!(node.outputs.len() == 1);
            for &output in &node.outputs {
                let t = &self.values[&output].tensor;
                size = t.dims().total_elems() * t.elem_ty().size();
                outputs.push(self.values[&output].buf);
            }
            let max_size: usize = self.device.max_work_item_sizes().unwrap().iter().product();
            assert!(size <= max_size);

            let kernel_event = unsafe {
                ExecuteKernel::new(&plan.kernel)
                    .set_arg(&outputs[0])
                    .set_arg(&inputs[0])
                    .set_arg(&inputs[1])
                    .set_global_work_size(size)
                    .enqueue_nd_range(&self.queue)
            }
            .unwrap();
            events.push(kernel_event);
        }

        let mut outputs = vec![];

        for out_id in &self.model.outputs {
            let val = &self.values[out_id];
            let out = Tensor::uninit_of_type(val.tensor.elem_ty(), val.tensor.dims().clone());

            let event = unsafe {
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
            };
            if let Err(e) = event {
                println!(">>>> {}", e.to_string());
                todo!();
            }

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
