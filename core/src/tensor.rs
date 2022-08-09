use crate::dim::{Dimension, Dimensions};

#[derive(Debug, Clone)]
pub struct Tensor {
    dims: Dimensions,
    stride: Dimensions,
    data: Vec<u8>,
    elem_ty: TensorElemType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorElemType {
    Bool,
    F32,
    I32,
    I64,
}

pub trait TensorElemTypeExt: PartialEq + PartialOrd + Copy {
    fn get_type() -> TensorElemType;
    fn zero() -> Self;
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
            TensorElemType::Bool => Self::new(dims, vec![0u8; total_elems]),
            TensorElemType::F32 => Self::new(dims, vec![0.0f32; total_elems]),
            TensorElemType::I32 => Self::new(dims, vec![0i32; total_elems]),
            TensorElemType::I64 => Self::new(dims, vec![0i64; total_elems]),
        }
    }

    pub fn set_raw_vec<T>(&mut self, data: Vec<T>) {
        let data = std::mem::ManuallyDrop::new(data);
        self.data = unsafe {
            Vec::from_raw_parts(
                data.as_ptr() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
                data.capacity() * std::mem::size_of::<T>(),
            )
        };
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

    pub fn fixed_dims<const N: usize>(&self) -> [Dimension; N] {
        let mut dims: [Dimension; N] = [0; N];
        dims.copy_from_slice(self.dims.as_slice());
        dims
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

    pub fn elem_ty(&self) -> &TensorElemType {
        &self.elem_ty
    }

    pub fn strides(&self) -> &[Dimension] {
        self.stride.as_slice()
    }

    pub fn verify(&self) -> bool {
        self.data.len() / self.elem_ty.size() == self.dims.total_elems()
    }
}

impl TensorElemType {
    pub fn size(&self) -> usize {
        match self {
            TensorElemType::Bool => std::mem::size_of::<u8>(),
            TensorElemType::F32 => std::mem::size_of::<f32>(),
            TensorElemType::I32 => std::mem::size_of::<i32>(),
            TensorElemType::I64 => std::mem::size_of::<i64>(),
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    pub fn is_f32(&self) -> bool {
        matches!(self, Self::F32)
    }

    pub fn is_i32(&self) -> bool {
        matches!(self, Self::I32)
    }

    pub fn is_i64(&self) -> bool {
        matches!(self, Self::I64)
    }
}

impl TensorElemTypeExt for u8 {
    fn get_type() -> TensorElemType {
        TensorElemType::Bool
    }

    fn zero() -> Self {
        0
    }
}

impl TensorElemTypeExt for bool {
    fn get_type() -> TensorElemType {
        TensorElemType::Bool
    }

    fn zero() -> Self {
        false
    }
}

impl TensorElemTypeExt for f32 {
    fn get_type() -> TensorElemType {
        TensorElemType::F32
    }

    fn zero() -> Self {
        0f32
    }
}

impl TensorElemTypeExt for i32 {
    fn get_type() -> TensorElemType {
        TensorElemType::I32
    }

    fn zero() -> Self {
        0i32
    }
}

impl TensorElemTypeExt for i64 {
    fn get_type() -> TensorElemType {
        TensorElemType::I64
    }

    fn zero() -> Self {
        0i64
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

#[test]
fn test_zeros() {
    fn all_zero<T: TensorElemTypeExt>(slice: &[T]) -> bool {
        slice.iter().all(|&x| x == T::zero())
    }
    let zeros_bool = Tensor::zeros::<bool>(vec![1, 1, 28, 28].into());
    let zeros_f32 = Tensor::zeros::<f32>(vec![1, 1, 28, 28].into());
    let zeros_i32 = Tensor::zeros::<i32>(vec![1, 1, 28, 28].into());
    let zeros_i64 = Tensor::zeros::<i64>(vec![1, 1, 28, 28].into());
    assert!(all_zero(zeros_bool.data::<bool>()));
    assert!(all_zero(zeros_f32.data::<f32>()));
    assert!(all_zero(zeros_i32.data::<i32>()));
    assert!(all_zero(zeros_i64.data::<i64>()));
}
