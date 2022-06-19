use crate::dim::{Dimension, Dimensions};

#[derive(Debug, Clone)]
pub struct Tensor {
    dims: Dimensions,
    stride: Dimensions,
    data: Vec<u8>,
    elem_ty: TensorElemType,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorElemType {
    F32,
    I64,
}

pub trait TensorElemTypeExt {
    fn get_type() -> TensorElemType;
}

impl Tensor {
    pub fn new<T: TensorElemTypeExt>(dims: Dimensions, data: Vec<T>) -> Self {
        let data = std::mem::ManuallyDrop::new(data);
        Self {
            stride: compute_strides(&dims),
            elem_ty: T::get_type(),
            data: unsafe {
                Vec::from_raw_parts(
                    data.as_ptr() as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                    data.capacity() * std::mem::size_of::<T>(),
                )
            },
            dims,
        }
    }

    pub fn new_from_raw(dims: Dimensions, elem_ty: TensorElemType, data: Vec<u8>) -> Self {
        let data = std::mem::ManuallyDrop::new(data);
        Self {
            stride: compute_strides(&dims),
            elem_ty,
            data: unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut u8, data.len(), data.capacity())
            },
            dims,
        }
    }

    pub fn zeros<T: TensorElemTypeExt>(dims: Dimensions) -> Self {
        let total_elems = dims.total_elems();
        match T::get_type() {
            TensorElemType::F32 => Self::new(dims, vec![0.0f32; total_elems]),
            TensorElemType::I64 => Self::new(dims, vec![0i64; total_elems]),
        }
    }

    pub fn reshape_into(mut self, dims: Dimensions) -> Self {
        self.stride = compute_strides(&dims);
        self.dims = dims;
        assert!(self.verify());
        self
    }

    pub fn to_transposed_2d(&self) -> Self {
        assert!(self.dims.len() == 2);
        let mut out = Tensor::zeros::<f32>(vec![self.dims[1], self.dims[0]].into());
        for x in 0..self.dims[0] {
            for y in 0..self.dims[1] {
                *out.at_2d_mut(y, x) = self.at_2d(x, y);
            }
        }
        out
    }

    pub fn at(&self, indices: &[Dimension]) -> f32 {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        self.data::<f32>()[index]
    }

    pub fn at_mut(&mut self, indices: &[Dimension]) -> &mut f32 {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        &mut self.data_mut::<f32>()[index]
    }

    pub fn at_2d(&self, x: Dimension, y: Dimension) -> f32 {
        self.data::<f32>()[self.stride[0] * x + self.stride[1] * y]
    }

    pub fn at_2d_mut(&mut self, x: Dimension, y: Dimension) -> &mut f32 {
        let offset = self.stride[0] * x + self.stride[1] * y;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn at_3d(&self, x: Dimension, y: Dimension, z: Dimension) -> f32 {
        self.data::<f32>()[self.stride[0] * x + self.stride[1] * y + self.stride[2] * z]
    }

    pub fn at_3d_mut(&mut self, x: Dimension, y: Dimension, z: Dimension) -> &mut f32 {
        let offset = self.stride[0] * x + self.stride[1] * y + self.stride[2] * z;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn at_4d(&self, x: Dimension, y: Dimension, z: Dimension, u: Dimension) -> f32 {
        self.data::<f32>()
            [self.stride[0] * x + self.stride[1] * y + self.stride[2] * z + self.stride[3] * u]
    }

    pub fn at_4d_mut(
        &mut self,
        x: Dimension,
        y: Dimension,
        z: Dimension,
        u: Dimension,
    ) -> &mut f32 {
        let offset =
            self.stride[0] * x + self.stride[1] * y + self.stride[2] * z + self.stride[3] * u;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn dims(&self) -> &Dimensions {
        &self.dims
    }

    pub fn data<T>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                self.data.len() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn data_mut<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_ptr() as *mut T,
                self.data.len() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn verify(&self) -> bool {
        self.data.len() / self.elem_ty.size() == self.dims.total_elems()
    }
}

impl TensorElemType {
    pub fn size(&self) -> usize {
        match self {
            TensorElemType::F32 => std::mem::size_of::<f32>(),
            TensorElemType::I64 => std::mem::size_of::<i64>(),
        }
    }
}

impl TensorElemTypeExt for f32 {
    fn get_type() -> TensorElemType {
        TensorElemType::F32
    }
}

impl TensorElemTypeExt for i64 {
    fn get_type() -> TensorElemType {
        TensorElemType::I64
    }
}

fn compute_strides(dims: &Dimensions) -> Dimensions {
    let mut strides = vec![];
    for i in 0..dims.len() {
        strides.push(dims[i + 1..].iter().product());
    }
    strides.into()
}

#[test]
fn create_tensors() {
    let _ = Tensor::zeros::<f32>(Dimensions(vec![1, 1, 28, 28]));
    let t = Tensor::new(
        vec![4, 4].into(),
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
    );
    assert!(t.verify());
}
