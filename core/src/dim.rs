pub type Dimension = usize;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dimensions(pub Vec<Dimension>);

impl Dimensions {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn total_elems(&self) -> usize {
        if self.len() == 0 {
            return 0;
        }
        let mut elems = 1;
        for d in &self.0 {
            elems *= d;
        }
        elems
    }

    pub fn as_slice(&self) -> &[Dimension] {
        self.0.as_slice()
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
