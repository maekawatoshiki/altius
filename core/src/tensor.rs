use crate::dim::Dimensions;

#[derive(Default)]
pub struct Tensor {
    dims: Dimensions,
    data: TensorData,
}

#[derive(Default)]
pub struct TensorData {
    // elem_ty: Type,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(dims: Dimensions) -> Self {
        Self {
            dims,
            data: TensorData::new_empty(),
        }
    }

    pub fn with_data(mut self, data: TensorData) -> Self {
        self.data = data;
        self
    }

    pub fn verify(&self) -> bool {
        self.data.len() == self.dims.total_elems()
    }
}

impl TensorData {
    pub fn new_empty() -> Self {
        Self { data: vec![] }
    }

    pub fn new_raw(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[test]
fn create_tensors() {
    let _ = Tensor::new(Dimensions(vec![1, 1, 28, 28]));
    let t = Tensor::new(Dimensions(vec![4, 4])).with_data(TensorData::new_raw(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]));
    assert!(t.verify());
}
