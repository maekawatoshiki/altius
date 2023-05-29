use std::{
    fmt,
    ops::{Deref, Index, IndexMut},
    slice::SliceIndex,
};

use crate::dim::{FixedDimension, FixedDimensions};

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Fixed(FixedDimension),
    Dynamic(String),
}

/// An alternative to `FixedDimensions` that allows dynamic shape.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Dimensions(pub Vec<Dimension>);

impl Dimensions {
    pub const fn new(dims: Vec<Dimension>) -> Self {
        Self(dims)
    }

    pub fn is_fixed(&self) -> bool {
        self.0.iter().all(|d| matches!(d, Dimension::Fixed(_)))
    }

    pub fn is_dynamic(&self) -> bool {
        self.0.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
    }

    pub fn as_fixed_dims(&self) -> Option<FixedDimensions> {
        if self.is_dynamic() {
            return None;
        }

        Some(FixedDimensions(
            self.iter()
                .map(|d| match d {
                    Dimension::Fixed(d) => *d,
                    Dimension::Dynamic(_) => unreachable!(),
                })
                .collect(),
        ))
    }
}

impl AsRef<Dimensions> for Dimensions {
    fn as_ref(&self) -> &Dimensions {
        self
    }
}

impl<I> Index<I> for Dimensions
where
    I: SliceIndex<[Dimension]>,
{
    type Output = <I as SliceIndex<[Dimension]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<I> IndexMut<I> for Dimensions
where
    I: SliceIndex<[Dimension]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<Dimension>> for Dimensions {
    fn from(v: Vec<Dimension>) -> Dimensions {
        Dimensions(v)
    }
}

impl Deref for Dimensions {
    type Target = Vec<Dimension>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dimension::Fixed(d) => write!(f, "{}", d),
            Dimension::Dynamic(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Debug for Dimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[test]
fn use_symdims() {
    let _ = Dimensions(vec![
        Dimension::Dynamic("batch".into()),
        Dimension::Fixed(8),
    ]);
}
