use std::{cell::RefCell, cmp::Ordering, fmt, iter::Sum, mem::MaybeUninit, sync::Arc};

use crate::{
    dim::{Dimension, Dimensions},
    fixed_dim::{FixedDimension, FixedDimensions},
};
use ndarray::{CowArray, IxDyn};
use rand::{
    distributions::Standard, prelude::Distribution, rngs::StdRng, thread_rng, Rng, SeedableRng,
};

thread_local!(static RNG: RefCell<StdRng> =
    RefCell::new(StdRng::from_rng(thread_rng()).expect("Failed to seed StdRng.")));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    dims: FixedDimensions,
    stride: FixedDimensions,
    data: Arc<Vec<u8>>,
    elem_ty: TensorElemType,
}

/// Represents a type and shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedFixedShape {
    pub dims: FixedDimensions,
    pub elem_ty: TensorElemType,
}

/// Represents a type and shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedShape {
    pub dims: Dimensions,
    pub elem_ty: TensorElemType,
}

#[derive(Debug)]
pub struct Statistics<T>
where
    T: TensorElemTypeExt,
{
    pub max: T,
    pub min: T,
    pub mean: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorElemType {
    Bool,
    F32,
    I32,
    I64,
}

pub trait TensorElemTypeExt: PartialEq + PartialOrd + Copy {
    fn get_type() -> TensorElemType;
    fn zero() -> Self;
    fn close(a: Self, b: Self) -> bool;
}

impl Tensor {
    pub fn new<T: TensorElemTypeExt>(dims: FixedDimensions, data: Vec<T>) -> Self {
        let data = std::mem::ManuallyDrop::new(data);
        Self {
            stride: compute_strides(&dims),
            elem_ty: T::get_type(),
            data: Arc::new(unsafe {
                Vec::from_raw_parts(
                    data.as_ptr() as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                    data.capacity() * std::mem::size_of::<T>(),
                )
            }),
            dims,
        }
    }

    pub fn new_from_raw(dims: FixedDimensions, elem_ty: TensorElemType, data: Vec<u8>) -> Self {
        let data = std::mem::ManuallyDrop::new(data);
        Self {
            stride: compute_strides(&dims),
            elem_ty,
            data: Arc::new(unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut u8, data.len(), data.capacity())
            }),
            dims,
        }
    }

    pub fn zeros<T: TensorElemTypeExt>(dims: FixedDimensions) -> Self {
        let total_elems = dims.total_elems();
        Self::new(dims, vec![T::zero(); total_elems])
    }

    /// Returns `Tensor` of given type and shape but allocates no elements.
    /// This is used in shape inference.
    pub fn empty_of_type(ty: TensorElemType, dims: FixedDimensions) -> Self {
        match ty {
            TensorElemType::Bool => Self::new::<bool>(dims, vec![]),
            TensorElemType::F32 => Self::new::<f32>(dims, vec![]),
            TensorElemType::I32 => Self::new::<i32>(dims, vec![]),
            TensorElemType::I64 => Self::new::<i64>(dims, vec![]),
        }
    }

    pub fn zeros_of_type(ty: TensorElemType, dims: FixedDimensions) -> Self {
        let total_elems = dims.total_elems();
        match ty {
            TensorElemType::Bool => Self::new(dims, vec![0u8; total_elems]),
            TensorElemType::F32 => Self::new(dims, vec![0.0f32; total_elems]),
            TensorElemType::I32 => Self::new(dims, vec![0i32; total_elems]),
            TensorElemType::I64 => Self::new(dims, vec![0i64; total_elems]),
        }
    }

    pub fn uninit_of_type(ty: TensorElemType, dims: FixedDimensions) -> Self {
        fn uninit_of<T: TensorElemTypeExt>(total_elems: usize) -> Vec<T> {
            let mut vec = MaybeUninit::new(Vec::with_capacity(total_elems));
            unsafe {
                vec.assume_init_mut().set_len(total_elems);
                vec.assume_init()
            }
        }

        let total_elems = dims.total_elems();
        match ty {
            TensorElemType::Bool => Self::new(dims, uninit_of::<u8>(total_elems)),
            TensorElemType::F32 => Self::new(dims, uninit_of::<f32>(total_elems)),
            TensorElemType::I32 => Self::new(dims, uninit_of::<i32>(total_elems)),
            TensorElemType::I64 => Self::new(dims, uninit_of::<i64>(total_elems)),
        }
    }

    pub fn uninit<T: TensorElemTypeExt>(dims: FixedDimensions) -> Self {
        fn uninit_of<T: TensorElemTypeExt>(total_elems: usize) -> Vec<T> {
            let mut vec = MaybeUninit::new(Vec::with_capacity(total_elems));
            unsafe {
                vec.assume_init_mut().set_len(total_elems);
                vec.assume_init()
            }
        }

        let total_elems = dims.total_elems();
        Self::new(dims, uninit_of::<T>(total_elems))
    }

    pub fn rand<T>(dims: FixedDimensions) -> Self
    where
        T: TensorElemTypeExt,
        Standard: Distribution<T>,
    {
        let total_elems = dims.total_elems();
        Self::new(
            dims,
            RNG.with(|r| {
                (&mut *r.borrow_mut())
                    .sample_iter(Standard)
                    .take(total_elems)
                    .collect::<Vec<T>>()
            }),
        )
    }

    pub fn rand_of_type(ty: TensorElemType, dims: FixedDimensions) -> Self {
        fn new<T>(total_elems: usize) -> Vec<T>
        where
            T: TensorElemTypeExt,
            Standard: Distribution<T>,
        {
            RNG.with(|r| {
                (&mut *r.borrow_mut())
                    .sample_iter(Standard)
                    .take(total_elems)
                    .collect::<Vec<T>>()
            })
        }

        let total_elems = dims.total_elems();
        match ty {
            TensorElemType::Bool => Self::new(dims, new::<bool>(total_elems)),
            TensorElemType::F32 => Self::new(dims, new::<f32>(total_elems)),
            TensorElemType::I32 => Self::new(dims, new::<i32>(total_elems)),
            TensorElemType::I64 => Self::new(dims, new::<i64>(total_elems)),
        }
    }

    pub fn seed_rng_from_u64(seed: u64) {
        RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
    }

    pub fn set_raw_vec<T>(&mut self, data: Vec<T>) {
        let data = std::mem::ManuallyDrop::new(data);
        self.data = Arc::new(unsafe {
            Vec::from_raw_parts(
                data.as_ptr() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
                data.capacity() * std::mem::size_of::<T>(),
            )
        });
    }

    // pub fn reshape_into(mut self, dims: FixedDimensions) -> Self {
    //     self.stride = compute_strides(&dims);
    //     self.dims = dims;
    //     assert!(self.verify());
    //     self
    // }

    // pub fn to_transposed_2d(&self) -> Self {
    //     assert!(self.dims.len() == 2);
    //     let mut out = Tensor::zeros::<f32>(vec![self.dims[1], self.dims[0]].into());
    //     for x in 0..self.dims[0] {
    //         for y in 0..self.dims[1] {
    //             *out.at_2d_mut(y, x) = self.at_2d(x, y);
    //         }
    //     }
    //     out
    // }

    pub fn slice_at<T: TensorElemTypeExt>(&self, indices: &[FixedDimension]) -> &[T] {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        &self.data::<T>()[index..]
    }

    pub fn slice_at_mut<T: TensorElemTypeExt>(&mut self, indices: &[FixedDimension]) -> &mut [T] {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        &mut self.data_mut::<T>()[index..]
    }

    pub fn at(&self, indices: &[FixedDimension]) -> f32 {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        self.data::<f32>()[index]
    }

    pub fn at_mut(&mut self, indices: &[FixedDimension]) -> &mut f32 {
        let mut index = 0;
        for (idx, d) in indices.iter().zip(self.stride.as_slice().iter()) {
            index += d * idx;
        }
        &mut self.data_mut::<f32>()[index]
    }

    pub fn at_2d(&self, x: FixedDimension, y: FixedDimension) -> f32 {
        self.data::<f32>()[self.stride[0] * x + self.stride[1] * y]
    }

    pub fn at_2d_mut(&mut self, x: FixedDimension, y: FixedDimension) -> &mut f32 {
        let offset = self.stride[0] * x + self.stride[1] * y;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn at_3d(&self, x: FixedDimension, y: FixedDimension, z: FixedDimension) -> f32 {
        self.data::<f32>()[self.stride[0] * x + self.stride[1] * y + self.stride[2] * z]
    }

    pub fn at_3d_mut(
        &mut self,
        x: FixedDimension,
        y: FixedDimension,
        z: FixedDimension,
    ) -> &mut f32 {
        let offset = self.stride[0] * x + self.stride[1] * y + self.stride[2] * z;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn at_4d(
        &self,
        x: FixedDimension,
        y: FixedDimension,
        z: FixedDimension,
        u: FixedDimension,
    ) -> f32 {
        self.data::<f32>()
            [self.stride[0] * x + self.stride[1] * y + self.stride[2] * z + self.stride[3] * u]
    }

    pub fn at_4d_mut(
        &mut self,
        x: FixedDimension,
        y: FixedDimension,
        z: FixedDimension,
        u: FixedDimension,
    ) -> &mut f32 {
        let offset =
            self.stride[0] * x + self.stride[1] * y + self.stride[2] * z + self.stride[3] * u;
        &mut self.data_mut::<f32>()[offset]
    }

    pub fn dims(&self) -> &FixedDimensions {
        &self.dims
    }

    pub fn fixed_dims<const N: usize>(&self) -> [FixedDimension; N] {
        let mut dims: [FixedDimension; N] = [0; N];
        dims.copy_from_slice(self.dims.as_slice());
        dims
    }

    pub fn data<T: TensorElemTypeExt>(&self) -> &[T] {
        assert_eq!(self.elem_ty, T::get_type());
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                self.data.len() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn data_mut<T: TensorElemTypeExt>(&mut self) -> &mut [T] {
        assert_eq!(self.elem_ty, T::get_type());
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_ptr() as *mut T,
                self.data.len() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn data_as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    pub fn data_as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }

    pub fn data_as_bytes(&self) -> &[u8] {
        self.data.as_ref()
    }

    pub fn copy_data_from(&mut self, other: &Self) {
        self.data = other.data.clone()
    }

    pub fn elem_ty(&self) -> TensorElemType {
        self.elem_ty
    }

    pub fn strides(&self) -> &[FixedDimension] {
        self.stride.as_slice()
    }

    pub fn allclose<T: TensorElemTypeExt>(&self, other: &[T]) -> bool {
        if self.elem_ty != T::get_type() {
            return false;
        }

        let x = self.data::<T>();
        if x.len() != other.len() {
            return false;
        }

        x.iter().zip(other.iter()).all(|(&x, &y)| T::close(x, y))
    }

    pub fn verify(&self) -> bool {
        self.data.len() / self.elem_ty.size() == self.dims.total_elems()
    }

    pub fn strides_for_broadcasting(&self, dims: &[FixedDimension]) -> Option<FixedDimensions> {
        fn upcast(
            to: &[FixedDimension],
            from: &[FixedDimension],
            stride: &[FixedDimension],
        ) -> Option<FixedDimensions> {
            let mut new_stride = to.to_vec();

            if to.len() < from.len() {
                return None;
            }

            {
                let mut new_stride_iter = new_stride.iter_mut().rev();
                for ((er, es), dr) in from
                    .iter()
                    .rev()
                    .zip(stride.iter().rev())
                    .zip(new_stride_iter.by_ref())
                {
                    if *dr == *er {
                        *dr = *es;
                    } else if *er == 1 {
                        *dr = 0
                    } else {
                        return None;
                    }
                }

                for dr in new_stride_iter {
                    *dr = 0;
                }
            }
            Some(new_stride.into())
        }

        upcast(dims, self.dims.as_slice(), self.stride.as_slice())
    }

    pub fn statistics<'a, T>(&'a self) -> Statistics<T>
    where
        T: TensorElemTypeExt + Sum<&'a T> + 'a + Into<f64>,
    {
        let data = self.data::<T>();
        Statistics {
            max: *data
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap_or(&T::zero()),
            min: *data
                .iter()
                .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap_or(&T::zero()),
            mean: data.iter().sum::<T>().into() / data.len() as f64,
        }
    }
}

impl TypedFixedShape {
    pub fn new(dims: FixedDimensions, elem_ty: TensorElemType) -> Self {
        Self { dims, elem_ty }
    }
}

impl TypedShape {
    pub fn new(dims: Dimensions, elem_ty: TensorElemType) -> Self {
        Self { dims, elem_ty }
    }
}

impl From<TypedFixedShape> for TypedShape {
    fn from(typed: TypedFixedShape) -> Self {
        Self {
            dims: Dimensions::new(typed.dims.0.into_iter().map(Dimension::Fixed).collect()),
            elem_ty: typed.elem_ty,
        }
    }
}

impl From<FixedDimensions> for Dimensions {
    fn from(dims: FixedDimensions) -> Self {
        Dimensions::new(dims.0.into_iter().map(Dimension::Fixed).collect())
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

    fn close(a: Self, b: Self) -> bool {
        a == b
    }
}

impl TensorElemTypeExt for bool {
    fn get_type() -> TensorElemType {
        TensorElemType::Bool
    }

    fn zero() -> Self {
        false
    }

    fn close(a: Self, b: Self) -> bool {
        a == b
    }
}

impl TensorElemTypeExt for f32 {
    fn get_type() -> TensorElemType {
        TensorElemType::F32
    }

    fn zero() -> Self {
        0f32
    }

    fn close(a: Self, b: Self) -> bool {
        let atol = 1e-5;
        let rtol = 1e-8;
        ((a - b).abs() <= (atol + rtol * b.abs()))
            || (a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive())
    }
}

impl TensorElemTypeExt for i32 {
    fn get_type() -> TensorElemType {
        TensorElemType::I32
    }

    fn zero() -> Self {
        0i32
    }

    fn close(a: Self, b: Self) -> bool {
        a == b
    }
}

impl TensorElemTypeExt for i64 {
    fn get_type() -> TensorElemType {
        TensorElemType::I64
    }

    fn zero() -> Self {
        0i64
    }

    fn close(a: Self, b: Self) -> bool {
        a == b
    }
}

fn compute_strides(dims: &FixedDimensions) -> FixedDimensions {
    let mut strides = vec![];
    for i in 0..dims.len() {
        strides.push(dims[i + 1..].iter().product());
    }
    strides.into()
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn dump<T: TensorElemTypeExt + fmt::Debug>(
            f: &mut fmt::Formatter<'_>,
            data: &[T],
        ) -> fmt::Result {
            const MAX_ELEMS: usize = 10;
            if data.len() > MAX_ELEMS {
                write!(f, "[")?;
                for e in data[0..MAX_ELEMS / 2].iter() {
                    write!(f, "{e:?}, ")?;
                }
                write!(f, "...")?;
                for e in data[data.len() - MAX_ELEMS / 2 - 1..].iter() {
                    write!(f, ", {e:?}")?;
                }
                write!(f, "]")
            } else {
                write!(f, "{data:?}")
            }
        }

        write!(f, "Tensor({:?}, {:?}, ", self.dims, self.elem_ty)?;
        match self.elem_ty {
            TensorElemType::F32 => dump(f, self.data::<f32>())?,
            TensorElemType::I32 => dump(f, self.data::<i32>())?,
            TensorElemType::I64 => dump(f, self.data::<i64>())?,
            TensorElemType::Bool => dump(f, self.data::<bool>())?,
        }
        write!(f, ")")
    }
}

impl<T: TensorElemTypeExt> From<&CowArray<'_, T, IxDyn>> for Tensor {
    fn from(arr: &CowArray<T, IxDyn>) -> Self {
        let dims = arr.shape().to_vec().into();
        let data = arr.iter().cloned().collect();
        Self::new(dims, data)
    }
}

macro_rules! dump_tensor {
    ($ty:ident, $name:ident) => {
        #[test]
        fn $name() {
            let t = Tensor::zeros::<$ty>(vec![2, 3, 4].into());
            insta::assert_snapshot!(t);
        }
    };
}

dump_tensor!(f32, dump_f32_tensor);
dump_tensor!(i32, dump_i32_tensor);
dump_tensor!(i64, dump_i64_tensor);
dump_tensor!(bool, dump_bool_tensor);

#[test]
fn create_tensors() {
    assert!(Tensor::zeros::<u8>(FixedDimensions(vec![1, 1, 28, 28])).verify());
    assert!(Tensor::zeros::<f32>(FixedDimensions(vec![1, 1, 28, 28])).verify());
    assert!(Tensor::zeros::<i32>(FixedDimensions(vec![1, 1, 28, 28])).verify());
    assert!(Tensor::zeros::<i64>(FixedDimensions(vec![1, 1, 28, 28])).verify());
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

#[test]
fn test_tensor_elem_type() {
    assert_eq!(TensorElemType::Bool.size(), 1);
    assert_eq!(TensorElemType::F32.size(), 4);
    assert_eq!(TensorElemType::I32.size(), 4);
    assert_eq!(TensorElemType::I64.size(), 8);
    assert!(TensorElemType::Bool.is_bool());
    assert!(TensorElemType::F32.is_f32());
    assert!(TensorElemType::I32.is_i32());
    assert!(TensorElemType::I64.is_i64());
}

#[test]
fn test_tensor_elem_type_ext() {
    assert!(<u8 as TensorElemTypeExt>::get_type().is_bool());
    assert!(<f32 as TensorElemTypeExt>::get_type().is_f32());
    assert!(<i32 as TensorElemTypeExt>::get_type().is_i32());
    assert!(<i64 as TensorElemTypeExt>::get_type().is_i64());
}

#[test]
fn test_tensor_rand() {
    fn check<T>()
    where
        T: TensorElemTypeExt,
        Standard: Distribution<T>,
    {
        let shape = vec![3, 6, 2, 9, 1, 11];
        let x = Tensor::rand::<T>(shape.to_owned().into());
        let y = Tensor::rand::<T>(shape.into());
        assert_ne!(x, y);
    }

    check::<f32>();
    check::<i32>();
    check::<i64>();
    check::<bool>();
}

#[test]
fn test_tensor_rand_of_type() {
    use TensorElemType::*;

    fn check(ty: TensorElemType) {
        let shape = vec![3, 6, 2, 9, 1, 11];
        let x = Tensor::rand_of_type(ty, shape.to_owned().into());
        let y = Tensor::rand_of_type(ty, shape.into());
        assert_ne!(x, y);
    }

    check(F32);
    check(I32);
    check(I64);
    check(Bool);
}
