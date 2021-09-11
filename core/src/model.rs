use crate::node::{NodeArena, NodeBuilder, NodeId};

pub struct Model {
    nodes: NodeArena,
    input_node: Option<NodeId>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            nodes: NodeArena::new(),
            input_node: None,
        }
    }
}

impl NodeBuilder for Model {
    fn arena(&self) -> &NodeArena {
        &self.nodes
    }

    fn arena_mut(&mut self) -> &mut NodeArena {
        &mut self.nodes
    }
}

#[test]
fn create_model() {
    use crate::node::*;
    let mut m = Model::new();
    let conv = m.new(
        Conv2d {
            input_dims: vec![1, 1, 28, 28].into(),
            weight_dims: vec![8, 1, 5, 5].into(),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            ..Default::default()
        }
        .into(),
    );
    let add = m.new(
        Add {
            input_a_dims: vec![1, 8, 28, 28].into(),
            input_b_dims: vec![8, 1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            ..Default::default()
        }
        .into(),
    );
    let relu = m.new(
        Relu {
            input_dims: vec![1, 8, 28, 28].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            ..Default::default()
        }
        .into(),
    );
    let max_pool = m.new(
        MaxPool {
            input_dims: vec![1, 8, 28, 28].into(),
            kernel: vec![2, 2].into(),
            stride: vec![2, 2].into(),
            output_dims: vec![1, 8, 14, 14].into(),
            ..Default::default()
        }
        .into(),
    );
    let conv2 = m.new(
        Conv2d {
            input_dims: vec![1, 8, 14, 14].into(),
            weight_dims: vec![16, 8, 5, 5].into(),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            ..Default::default()
        }
        .into(),
    );
    let add2 = m.new(
        Add {
            input_a_dims: vec![1, 16, 14, 14].into(),
            input_b_dims: vec![16, 1, 1].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            ..Default::default()
        }
        .into(),
    );
    let relu2 = m.new(
        Relu {
            input_dims: vec![1, 16, 14, 14].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            ..Default::default()
        }
        .into(),
    );
    let max_pool2 = m.new(
        MaxPool {
            input_dims: vec![1, 16, 14, 14].into(),
            kernel: vec![3, 3].into(),
            stride: vec![3, 3].into(),
            output_dims: vec![1, 16, 4, 4].into(),
            ..Default::default()
        }
        .into(),
    );
}
