use std::fmt;

use crate::dim::FixedDimension;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Fixed(FixedDimension),
    Dynamic(String),
}

/// An alternative to `FixedDimensions` that allows dynamic shape.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Dimensions(Vec<Dimension>);

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
