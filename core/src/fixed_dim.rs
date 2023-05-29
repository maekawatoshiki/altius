use std::{
    ops::{Deref, Index, IndexMut},
    slice::SliceIndex,
};

pub type FixedDimension = usize;

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct FixedDimensions(pub Vec<FixedDimension>);

impl std::fmt::Debug for FixedDimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl FixedDimensions {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_scalar(&self) -> bool {
        self.is_empty() || (self.len() == 1 && matches!(self.0[0], 0 | 1))
    }

    pub fn total_elems(&self) -> usize {
        self.0.iter().product()
    }

    pub fn as_slice(&self) -> &[FixedDimension] {
        self.0.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [FixedDimension] {
        self.0.as_mut_slice()
    }

    pub fn to_i32_vec(&self) -> Vec<i32> {
        self.0.iter().map(|&x| x as i32).collect()
    }

    pub fn from_i64(dims: &[i64]) -> Self {
        Self(dims.iter().map(|&x| x as FixedDimension).collect())
    }

    pub fn broadcast(&self, other: impl AsRef<Self>) -> Option<Self> {
        broadcast(&[self, other.as_ref()])
    }

    pub fn to_fixed_dims<const N: usize>(&self) -> [FixedDimension; N] {
        let mut dims: [FixedDimension; N] = [0; N];
        dims.copy_from_slice(&self.0);
        dims
    }

    pub fn strides(&self) -> Self {
        compute_strides(self)
    }

    pub fn strides_for_broadcasting_to(&self, dims: &[FixedDimension]) -> Option<FixedDimensions> {
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

        upcast(dims, self, compute_strides(self).as_slice())
    }
}

fn compute_strides(dims: &FixedDimensions) -> FixedDimensions {
    let mut strides = vec![];
    for i in 0..dims.len() {
        strides.push(dims[i + 1..].iter().product());
    }
    strides.into()
}

pub fn broadcast(shapes: &[impl AsRef<FixedDimensions>]) -> Option<FixedDimensions> {
    let mut shape = vec![];
    let max_len = shapes
        .iter()
        .map(AsRef::as_ref)
        .map(FixedDimensions::len)
        .max()?;
    for i in 0..max_len {
        let mut size = 1;
        for shape in shapes.iter().map(AsRef::as_ref) {
            let len = shape.len();
            let dim = if i < len { shape[len - i - 1] } else { 1 };
            if dim == 1 {
                continue;
            }
            if size != 1 && dim != size {
                return None;
            }
            size = dim
        }
        shape.push(size)
    }
    shape.reverse();
    Some(shape.into())
}

impl AsRef<FixedDimensions> for FixedDimensions {
    fn as_ref(&self) -> &FixedDimensions {
        self
    }
}

impl<I> Index<I> for FixedDimensions
where
    I: SliceIndex<[FixedDimension]>,
{
    type Output = <I as SliceIndex<[FixedDimension]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<I> IndexMut<I> for FixedDimensions
where
    I: SliceIndex<[FixedDimension]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<FixedDimension>> for FixedDimensions {
    fn from(v: Vec<FixedDimension>) -> FixedDimensions {
        FixedDimensions(v)
    }
}

impl Deref for FixedDimensions {
    type Target = Vec<usize>;
    fn deref(&self) -> &Vec<usize> {
        &self.0
    }
}

#[test]
fn total_elems() {
    assert_eq!(FixedDimensions(vec![1, 1, 28, 28]).total_elems(), 784)
}

#[test]
fn total_elems2() {
    assert_eq!(FixedDimensions(vec![1]).total_elems(), 1)
}

#[test]
fn total_elems3() {
    assert_eq!(FixedDimensions(vec![]).total_elems(), 1)
}

#[test]
fn broadcast_1() {
    let one = FixedDimensions::from(vec![1]);
    let shape = broadcast(&[&one]).unwrap();
    assert_eq!(shape, one)
}

#[test]
fn broadcast_2() {
    let one = FixedDimensions::from(vec![1]);
    let four = FixedDimensions::from(vec![4, 1]);
    let shape = broadcast(&[one, four]).unwrap();
    assert_eq!(shape, vec![4, 1].into())
}

#[test]
fn broadcast_3() {
    let one = FixedDimensions::from(vec![1]);
    let four = FixedDimensions::from(vec![4, 1]);
    let shape = broadcast(&[four, one]).unwrap();
    assert_eq!(shape, vec![4, 1].into())
}

#[test]
#[should_panic]
fn broadcast_4() {
    let one = FixedDimensions::from(vec![10, 20]);
    let four = FixedDimensions::from(vec![10, 20, 30]);
    let _ = broadcast(&[four, one]).unwrap();
}

#[test]
fn broadcast_5() {
    let x = FixedDimensions::from(vec![1, 3, 3]);
    let y = FixedDimensions::from(vec![5, 1, 3, 3]);
    let shape = broadcast(&[x, y]).unwrap();
    assert_eq!(shape, vec![5, 1, 3, 3].into())
}

#[test]
fn broadcast_6() {
    let x = FixedDimensions::from(vec![1, 3, 1]);
    let y = FixedDimensions::from(vec![5, 3, 10]);
    let shape = broadcast(&[x, y]).unwrap();
    assert_eq!(shape, vec![5, 3, 10].into())
}

#[test]
fn broadcast_7() {
    let x = FixedDimensions::from(vec![1, 3, 4, 4]);
    let y = FixedDimensions::from(vec![3, 1, 1]);
    let shape = broadcast(&[x, y]).unwrap();
    assert_eq!(shape, vec![1, 3, 4, 4,].into())
}
