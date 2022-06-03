use crate::{
    node::{Node2Arena, Node2Id, NodeArena, NodeBuilder, NodeId},
    tensor::Tensor2,
    value::{ValueArena, ValueId},
};
use rustc_hash::{FxHashMap, FxHashSet};

pub struct Model {
    nodes: NodeArena,
    pub input_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Model2 {
    pub nodes: Node2Arena,
    pub values: ValueArena,
    pub inits: FxHashMap<ValueId, Tensor2>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Model2 {
    pub fn get_value_users(&self) -> FxHashMap<ValueId, FxHashSet<Node2Id>> {
        let mut value_users: FxHashMap<ValueId, FxHashSet<Node2Id>> = FxHashMap::default();

        for (node_id, node) in self.nodes.iter() {
            for &input in node.inputs.iter() {
                value_users.entry(input).or_default().insert(node_id);
            }
        }

        value_users
    }

    pub fn topo_sort_nodes(&self) -> Vec<Node2Id> {
        let value_users = self.get_value_users();

        let mut nodes = vec![];
        let mut num_node_inputs = FxHashMap::default();
        let mut que = vec![];

        let mut consts = self.inits.keys().copied().collect::<FxHashSet<_>>();
        consts.insert(self.inputs[0]);
        for (id, node) in self.nodes.iter() {
            let inputs = &node.inputs.clone().into_iter().collect::<FxHashSet<_>>() - &consts;
            num_node_inputs.insert(id, inputs.len());
            if inputs.is_empty() {
                que.push(id);
            }
        }

        while let Some(id) = que.pop() {
            nodes.push(id);
            for output in self.nodes[id].outputs.iter() {
                if self.outputs.contains(output) {
                    continue;
                }
                for n in value_users[output].iter() {
                    *num_node_inputs.get_mut(n).unwrap() -= 1;
                    if *num_node_inputs.get(n).unwrap() == 0 {
                        que.push(*n);
                    }
                }
            }
        }

        nodes
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
// #[allow(unused_variables)]
fn mnist_model2() {
    use crate::node::{Node2, Op};

    let mut m = Model2::default();

    let conv0_in = m.values.new_val(); // Input tensor [1, 1, 28, 28]
    let conv0_weight = m.values.new_val();
    let conv0_out = m.values.new_val();
    let _conv0 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![].into())
        .with_in(conv0_in)
        .with_in(conv0_weight)
        .with_out(conv0_out)
        .alloc(&mut m.nodes);

    let add0_const = m.values.new_val();
    let add0_out = m.values.new_val();
    let _add0 = Node2::new(Op::Add)
        .with_in(conv0_out)
        .with_in(add0_const)
        .with_out(add0_out)
        .alloc(&mut m.nodes);

    let relu0_out = m.values.new_val();
    let _relu0 = Node2::new(Op::ReLU)
        .with_in(add0_out)
        .with_out(relu0_out)
        .alloc(&mut m.nodes);

    let maxpool0_out = m.values.new_val();
    let _maxpool0 = Node2::new(Op::MaxPool)
        .with_attr(vec![2, 2].into())
        .with_attr(vec![2, 2].into())
        .with_in(relu0_out)
        .with_out(maxpool0_out)
        .alloc(&mut m.nodes);

    let conv1_weight = m.values.new_val();
    let conv1_out = m.values.new_val();
    let _conv1 = Node2::new(Op::Conv2d)
        .with_attr(vec![5, 5].into())
        .with_attr(vec![1, 1].into())
        .with_attr(vec![2, 2].into())
        .with_in(maxpool0_out)
        .with_in(conv1_weight)
        .with_out(conv1_out)
        .alloc(&mut m.nodes);

    let add1_const = m.values.new_val();
    let add1_out = m.values.new_val();
    let _add1 = Node2::new(Op::Add)
        .with_in(conv1_out)
        .with_in(add1_const)
        .with_out(add1_out)
        .alloc(&mut m.nodes);

    let relu1_out = m.values.new_val();
    let _relu1 = Node2::new(Op::ReLU)
        .with_in(add1_out)
        .with_out(relu1_out)
        .alloc(&mut m.nodes);

    let maxpool1_out = m.values.new_val();
    let _maxpool1 = Node2::new(Op::MaxPool)
        .with_in(relu1_out)
        .with_out(maxpool1_out)
        .with_attr(vec![3, 3].into())
        .with_attr(vec![3, 3].into())
        .alloc(&mut m.nodes);

    let reshape0_const = m.values.new_val();
    let reshape0_out = m.values.new_val();
    let _reshape0 = Node2::new(Op::Reshape)
        .with_in(maxpool1_out)
        .with_in(reshape0_const)
        .with_out(reshape0_out)
        .alloc(&mut m.nodes);

    let reshape1_const0 = m.values.new_val();
    let reshape1_const1 = m.values.new_val();
    let reshape1_out = m.values.new_val();
    let _reshape1 = Node2::new(Op::Reshape)
        .with_in(reshape1_const0)
        .with_in(reshape1_const1)
        .with_out(reshape1_out)
        .alloc(&mut m.nodes);

    let matmul0_out = m.values.new_val();
    let _matmul0 = Node2::new(Op::MatMul)
        .with_in(reshape0_out)
        .with_in(reshape1_out)
        .with_out(matmul0_out)
        .alloc(&mut m.nodes);

    let add2_const = m.values.new_val();
    let add2_out = m.values.new_val();
    let _add2 = Node2::new(Op::Add)
        .with_in(matmul0_out)
        .with_in(add2_const)
        .with_out(add2_out)
        .alloc(&mut m.nodes);

    m.inputs.push(conv0_in);
    m.outputs.push(add2_out);

    m.inits
        .insert(add0_const, Tensor2::new(vec![8, 1, 5, 5].into()));
    m.inits
        .insert(add1_const, Tensor2::new(vec![8, 1, 1].into()));
    m.inits
        .insert(add2_const, Tensor2::new(vec![16, 1, 1].into()));
    m.inits
        .insert(conv0_weight, Tensor2::new(vec![8, 1, 5, 5].into()));
    m.inits
        .insert(conv1_weight, Tensor2::new(vec![16, 8, 5, 5].into()));
    m.inits.insert(
        reshape0_const,
        Tensor2::new(vec![2].into()).with_data(vec![1, 256].into()),
    );
    m.inits
        .insert(reshape1_const0, Tensor2::new(vec![16, 4, 4, 10].into()));
    m.inits.insert(
        reshape1_const1,
        Tensor2::new(vec![2].into()).with_data(vec![256, 10].into()),
    );

    let order = m.topo_sort_nodes();
    // println!(
    //     "{:#?}",
    //     order.iter().map(|&n| m.nodes[n].op).collect::<Vec<_>>()
    // );
    insta::assert_debug_snapshot!(order);
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
