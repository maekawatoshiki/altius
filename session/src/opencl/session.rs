use altius_core::model::Model;
use opencl3::{command_queue::CommandQueue, context::Context, device::Device};

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
}

impl<'a> OpenclSession<'a> {}
