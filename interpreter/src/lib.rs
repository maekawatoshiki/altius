use altius_core::{
    model::Model,
    node::{Add, MatMul, Node, NodeBuilder, NodeId},
    tensor::Tensor,
};

pub struct Interpreter<'a> {
    model: &'a Model,
}

impl<'a> Interpreter<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self { model }
    }

    pub fn run(&mut self, input: Tensor) -> Tensor {
        todo!()
    }

    pub fn run_node(&mut self, id: NodeId) -> Tensor {
        let node = &self.model.arena()[id];
        match node {
            Node::Conv2d(conv2d) => {
                todo!()
            }
            Node::Reshape(node) => self.run_node_reshape(node),
            Node::MatMul(node) => self.run_node_mat_mul(node),
            Node::Add(node) => self.run_node_add(node),
            _ => todo!(),
        }
    }

    fn run_node_reshape(&mut self, node: &MatMul) -> Tensor {
        todo!()
    }

    fn run_node_mat_mul(&mut self, node: &MatMul) -> Tensor {
        let input_a = self.run_node(node.input_a_node.unwrap());
        let input_b = self.run_node(node.input_b_node.unwrap());

        assert!(node.input_a_dims.len() == 2);
        assert!(node.input_b_dims.len() == 2);
        assert!(node.input_a_dims.as_slice()[1] == node.input_b_dims.as_slice()[0]);

        let mut output = Tensor::new(node.output_dims.clone());
        for i in 0..input_a.dims().as_slice()[0] {
            for j in 0..input_b.dims().as_slice()[1] {
                let mut t = 0.0;
                for k in 0..input_b.dims().as_slice()[0] {
                    t += input_a.at(&[i, k]) * input_b.at(&[k, j]);
                }
                *output.at_mut(&[i, j]) = t;
            }
        }
        output
    }

    fn run_node_add(&mut self, node: &Add) -> Tensor {
        let input_a = self.run_node(node.input_a_node.unwrap());
        let input_b = self.run_node(node.input_b_node.unwrap());

        if node.input_a_dims == node.input_b_dims {
            let mut output = Tensor::new(node.output_dims.clone());
            for (i, (a, b)) in input_a.data().iter().zip(input_b.data().iter()).enumerate() {
                output.data_mut()[i] = a + b;
            }
            return output;
        }

        todo!()
    }
}

#[test]
fn run() {
    let mnist = mnist();
}

#[cfg(test)]
fn mnist() -> Model {
    use altius_core::{node::*, tensor::*};
    let mut m = Model::new();
    let input = m.new(Node::Input);
    let conv_weight = m.new(
        Tensor::new(vec![8, 1, 5, 5].into())
            .with_data(include!("../../core/examples/conv1").into())
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
            input_node: Some(input),
            ..Default::default()
        }
        .into(),
    );
    // m.input_node = Some(conv);
    let add_input_b = m.new(
        Tensor::new(vec![8, 1, 1].into())
            .with_data(
                include!("../../core/examples/add1")
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
                include!("../../core/examples/conv2")
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
                include!("../../core/examples/add2")
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
                include!("../../core/examples/reshape1")
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
                include!("../../core/examples/add3")
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
    m
}
