// TODO: Integrate core::tensor::Tensor

use altius_core::tensor::TypedShape;
use opencl3::types::cl_mem;

pub struct OpenclTensor {
    /// The type and shape of the tensor
    #[allow(dead_code)]
    shape: TypedShape,

    /// OpenCL buffer
    #[allow(dead_code)]
    buf: cl_mem,
}
