use crate::{
    node::{Node2Arena, Node2Id, NodeArena, NodeBuilder, NodeId},
    value::ValueArena,
};

pub struct Model {
    nodes: NodeArena,
    pub input_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Model2 {
    pub nodes: Node2Arena,
    pub values: ValueArena,
    pub inputs: Vec<Node2Id>,
    pub outputs: Vec<Node2Id>,
}

impl Model2 {
    pub fn wire(&mut self, from: Node2Id, to: Node2Id) {
        // self.nodes[from].outputs.push(to); self.nodes[to].inputs.push(from);
    }
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
#[allow(unused_variables)]
fn mnist_model2() {
    use crate::node::{Node2, Op};

    let mut m = Model2::default();

    let input = Node2::new(Op::Input)
        .with_attr(vec![1, 1, 28, 28].into())
        .alloc(&mut m.nodes);
    let conv0 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![].into())
        .alloc(&mut m.nodes);
    let add0 = Node2::new(Op::Add).alloc(&mut m.nodes);
    let relu0 = Node2::new(Op::ReLU).alloc(&mut m.nodes);
    let maxpool0 = Node2::new(Op::MaxPool)
        .with_attr(vec![2, 2].into())
        .with_attr(vec![2, 2].into())
        .alloc(&mut m.nodes);
    let conv1 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![2, 2].into())
        .alloc(&mut m.nodes);
    let add1 = Node2::new(Op::Add).alloc(&mut m.nodes);
    let relu1 = Node2::new(Op::ReLU).alloc(&mut m.nodes);
    let maxpool1 = Node2::new(Op::MaxPool)
        .with_attr(vec![3, 3].into())
        .with_attr(vec![3, 3].into())
        .alloc(&mut m.nodes);
    let reshape0 = Node2::new(Op::Reshape).alloc(&mut m.nodes);
    let reshape1 = Node2::new(Op::Reshape).alloc(&mut m.nodes);
    let matmul0 = Node2::new(Op::MatMul).alloc(&mut m.nodes);
    let add2 = Node2::new(Op::Add).alloc(&mut m.nodes);

    // m.add_input(conv0, input);
}

#[test]
fn mnist_model() {
    use crate::{node::*, tensor::*};
    let mut m = Model::new();
    let conv_weight = m.new(Tensor::new(vec![8, 1, 5, 5].into()).into());
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
    let add_input_b = m.new(Tensor::new(vec![8, 1, 1].into()).into());
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
    let relu = m.new_relu(add);
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
    let conv2_weight = m.new(Tensor::new(vec![16, 8, 5, 5].into()).into());
    let conv2 = m.new_conv2d(
        max_pool,
        conv2_weight,
        vec![5, 5].into(),
        vec![1, 1].into(),
        vec![2, 2].into(),
    );
    let add2_input_b = m.new(Tensor::new(vec![16, 1, 1].into()).into());
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
    let relu2 = m.new_relu(add2);
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
    let reshape2_input = m.new(Tensor::new(vec![16, 4, 4, 10].into()).into());
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
    let add3_input_b = m.new(Tensor::new(vec![1, 10].into()).into());
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
