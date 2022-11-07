use altius_core::{model::Model, value::ValueId};
use opencl3::{command_queue::CommandQueue, context::Context, device::Device};
use rustc_hash::FxHashMap;

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
}

impl<'a> OpenclSession<'a> {}
