// TODO: Integrate OpenclTensor with core::tensor::Tensor

use altius_core::tensor::Tensor;
use opencl3::types::cl_mem;

pub struct OpenclTensor {
    /// CPU Tensor
    #[allow(dead_code)]
    pub(super) tensor: Tensor,

    /// OpenCL buffer
    #[allow(dead_code)]
    pub(super) buf: cl_mem,
}
