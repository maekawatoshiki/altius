pub type Dimension = usize;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dimensions(pub Vec<Dimension>);

impl Dimensions {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn total_elems(&self) -> usize {
        self.0.iter().product()
    }

    pub fn as_slice(&self) -> &[Dimension] {
        self.0.as_slice()
    }

    pub fn from_i64(dims: &[i64]) -> Self {
        Self(dims.iter().map(|&x| x as Dimension).collect())
    }
}

impl From<Vec<Dimension>> for Dimensions {
    fn from(v: Vec<Dimension>) -> Dimensions {
        Dimensions(v)
    }
}

#[test]
fn total_elems() {
    assert_eq!(Dimensions(vec![1, 1, 28, 28]).total_elems(), 784)
}
