use crate::node::{NodeArena, NodeBuilder, NodeId};

pub struct Model {
    nodes: NodeArena,
    pub input_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            nodes: NodeArena::new(),
            input_node: None,
            output_node: None,
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
fn mnist_model() {
    use crate::{node::*, tensor::*};
    let mut m = Model::new();
    let conv_weight = m.new(
        Tensor::new(vec![8, 1, 5, 5].into())
            .with_data(include!("../examples/conv1").into())
            .into(),
    );
    let conv = m.new(
        Conv2d {
            input_dims: vec![1, 1, 28, 28].into(),
            weight_dims: vec![8, 1, 5, 5].into(),
            weight_node: Some(conv_weight),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            ..Default::default()
        }
        .into(),
    );
    // m.input_node = Some(conv);
    let add_input_b = m.new(
        Tensor::new(vec![8, 1, 1].into())
            .with_data(
                include!("../examples/add1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add = m.new(
        Add {
            input_a_dims: vec![1, 8, 28, 28].into(),
            input_b_dims: vec![8, 1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_a_node: Some(conv),
            input_b_node: Some(add_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu = m.new(
        Relu {
            input_dims: vec![1, 8, 28, 28].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(add),
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
            input_node: Some(relu),
            ..Default::default()
        }
        .into(),
    );
    let conv2_weight = m.new(
        Tensor::new(vec![16, 8, 5, 5].into())
            .with_data(
                include!("../examples/conv2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let conv2 = m.new(
        Conv2d {
            input_dims: vec![1, 8, 14, 14].into(),
            weight_dims: vec![16, 8, 5, 5].into(),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(max_pool),
            weight_node: Some(conv2_weight),
            ..Default::default()
        }
        .into(),
    );
    let add2_input_b = m.new(
        Tensor::new(vec![16, 1, 1].into())
            .with_data(
                include!("../examples/add2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add2 = m.new(
        Add {
            input_a_dims: vec![1, 16, 14, 14].into(),
            input_b_dims: vec![16, 1, 1].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_a_node: Some(conv2),
            input_b_node: Some(add2_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu2 = m.new(
        Relu {
            input_dims: vec![1, 16, 14, 14].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_node: Some(add2),
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
            input_node: Some(relu2),
            ..Default::default()
        }
        .into(),
    );
    let reshape = m.new(
        Reshape {
            input_dims: vec![1, 16, 4, 4].into(),
            output_dims: vec![1, 256].into(),
            input_node: Some(max_pool2),
            ..Default::default()
        }
        .into(),
    );
    let reshape2_input = m.new(
        Tensor::new(vec![16, 4, 4, 10].into())
            .with_data(
                include!("../examples/reshape1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let reshape2 = m.new(
        Reshape {
            input_dims: vec![16, 4, 4, 10].into(),
            output_dims: vec![256, 10].into(),
            input_node: Some(reshape2_input),
            ..Default::default()
        }
        .into(),
    );
    let mat_mal = m.new(
        MatMul {
            input_a_dims: vec![1, 256].into(),
            input_b_dims: vec![256, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(reshape),
            input_b_node: Some(reshape2),
            ..Default::default()
        }
        .into(),
    );
    let add3_input_b = m.new(
        Tensor::new(vec![1, 10].into())
            .with_data(
                include!("../examples/add3")
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add3 = m.new(
        Add {
            input_a_dims: vec![1, 10].into(),
            input_b_dims: vec![1, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(mat_mal),
            input_b_node: Some(add3_input_b),
            ..Default::default()
        }
        .into(),
    );
    m.output_node = Some(add3);
}
