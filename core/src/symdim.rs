use std::fmt;

use crate::dim::Dimension;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SymbolicDimension {
    Static(Dimension),
    Dynamic(String),
}

/// An alternative to `Dimensions` that allows dynamic shape.
/// This will replace `Dimensions` in the future.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SymbolicDimensions(Vec<SymbolicDimension>);

impl fmt::Debug for SymbolicDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicDimension::Static(d) => write!(f, "{}", d),
            SymbolicDimension::Dynamic(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Debug for SymbolicDimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[test]
fn use_symdims() {
    let _ = SymbolicDimensions(vec![
        SymbolicDimension::Dynamic("batch".into()),
        SymbolicDimension::Static(8),
    ]);
}
