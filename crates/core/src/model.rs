use std::{
    fmt::{self, Formatter},
    mem,
};

use crate::{
    analysis::shape::infer_shapes, fixed_dim::FixedDimensions, graph::Graph, node::NodeId,
    tensor::TensorElemType, value::ValueId,
};
use id_arena::Arena;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Default, Clone)]
pub struct Model {
    pub graph: Graph,
    pub opset_version: i64,
}

impl Model {
    pub fn lookup_named_value(&self, name: &str) -> Option<ValueId> {
        self.graph.values.inner().iter().find_map(|(id, value)| {
            if value.name.as_ref().is_some_and(|nm| nm == name) {
                Some(id)
            } else {
                None
            }
        })
    }

    pub fn get_value_users(&self) -> FxHashMap<ValueId, FxHashSet<NodeId>> {
        let mut value_users: FxHashMap<ValueId, FxHashSet<NodeId>> = FxHashMap::default();

        for (node_id, node) in self.graph.nodes.iter() {
            if node.deleted {
                continue;
            }

            for &input in node.inputs.iter() {
                value_users.entry(input).or_default().insert(node_id);
            }
        }

        value_users
    }

    pub fn get_value_parents(&self) -> FxHashMap<ValueId, NodeId> {
        let mut value_parents: FxHashMap<ValueId, NodeId> = FxHashMap::default();

        for (node_id, node) in self.graph.nodes.iter() {
            if node.deleted {
                continue;
            }

            for &output in node.outputs.iter() {
                value_parents.insert(output, node_id);
            }
        }

        value_parents
    }

    pub fn topo_sort_nodes(&self) -> Vec<NodeId> {
        let value_users = self.get_value_users();
        let value_parents = self.get_value_parents();

        let mut nodes = vec![];
        let mut num_node_inputs = FxHashMap::default();
        let mut que = vec![];

        let consts = &self.graph.inputs.iter().copied().collect::<FxHashSet<_>>()
            | &self.graph.inits.keys().copied().collect::<FxHashSet<_>>();
        for (id, node) in self.graph.nodes.iter() {
            if node.deleted {
                continue;
            }

            let mut inputs = &node.inputs.clone().into_iter().collect::<FxHashSet<_>>() - &consts;
            inputs.retain(|i| value_parents.contains_key(i)); // NOTE: Some of the node inputs might be optional (so they don't have their parents).
            num_node_inputs.insert(id, inputs.len());
            if inputs.is_empty() {
                que.push(id);
            }
        }

        while let Some(id) = que.pop() {
            nodes.push(id);
            for output in self.graph.nodes[id].outputs.iter() {
                if !value_users.contains_key(output) {
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

    /// Removes nodes labeled as deleted from the arena.
    pub fn remove_unnecessary_nodes(&mut self) {
        let order = self
            .topo_sort_nodes()
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect::<FxHashMap<NodeId, usize>>();
        let old_arena = mem::replace(&mut self.graph.nodes, Arena::new());
        let mut nodes = old_arena
            .into_iter()
            .filter(|(_, node)| !node.deleted)
            .collect::<Vec<_>>();
        nodes.sort_by(|x, y| order[&x.0].cmp(&order[&y.0]));
        for (_, old_node) in nodes {
            assert!(!old_node.deleted);
            self.graph.nodes.alloc(old_node);
        }
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[derive(Default)]
        struct DebugNode<'a> {
            op: &'static str,
            name: &'a str,
            inputs: Vec<(TensorElemType, &'a FixedDimensions)>,
            outputs: Vec<(TensorElemType, &'a FixedDimensions)>,
        }

        impl fmt::Debug for DebugNode<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}({}) in=(", self.op, self.name)?;
                for (i, (dtype, dims)) in self.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dims:?}:{dtype:?}")?;
                }
                write!(f, ") out=(")?;
                for (i, (dtype, dims)) in self.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dims:?}:{dtype:?}")?;
                }
                write!(f, ")")
            }
        }

        let mut value_shapes = FxHashMap::default();
        infer_shapes(self, &mut Default::default(), &mut value_shapes)
            .expect("Failed to infer shapes");

        let node_ids = self.topo_sort_nodes();
        let mut debug_nodes = vec![];
        for &node_id in &node_ids {
            let node = &self.graph.nodes[node_id];
            let mut debug_node = DebugNode {
                op: node.op.name(),
                name: node.name.as_ref().map_or("", |s| s),
                ..Default::default()
            };
            for &input in &node.inputs {
                let dtype = value_shapes[&input].elem_ty;
                let dims = &value_shapes[&input].dims;
                debug_node.inputs.push((dtype, dims));
            }
            for &output in &node.outputs {
                let dtype = value_shapes[&output].elem_ty;
                let dims = &value_shapes[&output].dims;
                debug_node.outputs.push((dtype, dims));
            }
            debug_nodes.push(debug_node);
        }

        f.debug_struct("Model")
            .field("graph", &debug_nodes)
            .field("opset_version", &self.opset_version)
            .finish()
    }
}

#[test]
fn mnist_model() {
    use crate::{
        node::Node,
        op::{Conv2d, MaxPool, Op},
        tensor::Tensor,
    };

    let mut m = Model::default();

    let conv0_in = m.graph.values.new_val(); // Input tensor [1, 1, 28, 28]
    let conv0_weight = m.graph.values.new_val();
    let conv0_out = m.graph.values.new_val();
    let _conv0 = Node::new(Op::Conv2d(Conv2d {
        auto_pad: "SAME_UPPER".into(),
        kernel_shape: vec![5, 5].into(),
        strides: vec![1, 1].into(),
        ..Default::default()
    }))
    .with_in(conv0_in)
    .with_in(conv0_weight)
    .with_out(conv0_out)
    .alloc(&mut m.graph.nodes);

    let add0_const = m.graph.values.new_val();
    let add0_out = m.graph.values.new_val();
    let _add0 = Node::new(Op::Add)
        .with_in(conv0_out)
        .with_in(add0_const)
        .with_out(add0_out)
        .alloc(&mut m.graph.nodes);

    let relu0_out = m.graph.values.new_val();
    let _relu0 = Node::new(Op::ReLU)
        .with_in(add0_out)
        .with_out(relu0_out)
        .alloc(&mut m.graph.nodes);

    let maxpool0_out = m.graph.values.new_val();
    let _maxpool0 = Node::new(Op::MaxPool(MaxPool {
        auto_pad: "NOTSET".into(),
        padding: vec![0, 0, 0, 0].into(),
        kernel_shape: vec![2, 2].into(),
        strides: vec![2, 2].into(),
    }))
    .with_in(relu0_out)
    .with_out(maxpool0_out)
    .alloc(&mut m.graph.nodes);

    let conv1_weight = m.graph.values.new_val();
    let conv1_out = m.graph.values.new_val();
    let _conv1 = Node::new(Op::Conv2d(Conv2d {
        auto_pad: "SAME_UPPER".into(),
        kernel_shape: vec![5, 5].into(),
        strides: vec![1, 1].into(),
        ..Default::default()
    }))
    .with_in(maxpool0_out)
    .with_in(conv1_weight)
    .with_out(conv1_out)
    .alloc(&mut m.graph.nodes);

    let add1_const = m.graph.values.new_val();
    let add1_out = m.graph.values.new_val();
    let _add1 = Node::new(Op::Add)
        .with_in(conv1_out)
        .with_in(add1_const)
        .with_out(add1_out)
        .alloc(&mut m.graph.nodes);

    let relu1_out = m.graph.values.new_val();
    let _relu1 = Node::new(Op::ReLU)
        .with_in(add1_out)
        .with_out(relu1_out)
        .alloc(&mut m.graph.nodes);

    let maxpool1_out = m.graph.values.new_val();
    let _maxpool1 = Node::new(Op::MaxPool(MaxPool {
        auto_pad: "NOTSET".into(),
        padding: vec![0, 0, 0, 0].into(),
        kernel_shape: vec![3, 3].into(),
        strides: vec![3, 3].into(),
    }))
    .with_in(relu1_out)
    .with_out(maxpool1_out)
    .alloc(&mut m.graph.nodes);

    let reshape0_const = m.graph.values.new_val();
    let reshape0_out = m.graph.values.new_val();
    let _reshape0 = Node::new(Op::Reshape)
        .with_in(maxpool1_out)
        .with_in(reshape0_const)
        .with_out(reshape0_out)
        .alloc(&mut m.graph.nodes);

    let reshape1_const0 = m.graph.values.new_val();
    let reshape1_const1 = m.graph.values.new_val();
    let reshape1_out = m.graph.values.new_val();
    let _reshape1 = Node::new(Op::Reshape)
        .with_in(reshape1_const0)
        .with_in(reshape1_const1)
        .with_out(reshape1_out)
        .alloc(&mut m.graph.nodes);

    let matmul0_out = m.graph.values.new_val();
    let _matmul0 = Node::new(Op::MatMul)
        .with_in(reshape0_out)
        .with_in(reshape1_out)
        .with_out(matmul0_out)
        .alloc(&mut m.graph.nodes);

    let add2_const = m.graph.values.new_val();
    let add2_out = m.graph.values.new_val();
    let _add2 = Node::new(Op::Add)
        .with_in(matmul0_out)
        .with_in(add2_const)
        .with_out(add2_out)
        .alloc(&mut m.graph.nodes);

    m.graph.inputs.push(conv0_in);
    m.graph.inputs.push(add0_const);
    m.graph.inputs.push(add1_const);
    m.graph.inputs.push(add2_const);
    m.graph.inputs.push(conv0_weight);
    m.graph.inputs.push(conv1_weight);
    m.graph.inputs.push(reshape0_const);
    m.graph.inputs.push(reshape1_const0);
    m.graph.inputs.push(reshape1_const1);
    m.graph.outputs.push(add2_out);

    m.graph
        .inits
        .insert(add0_const, Tensor::zeros::<f32>(vec![8, 1, 5, 5].into()));
    m.graph
        .inits
        .insert(add1_const, Tensor::zeros::<f32>(vec![8, 1, 1].into()));
    m.graph
        .inits
        .insert(add2_const, Tensor::zeros::<f32>(vec![16, 1, 1].into()));
    m.graph
        .inits
        .insert(conv0_weight, Tensor::zeros::<f32>(vec![8, 1, 5, 5].into()));
    m.graph
        .inits
        .insert(conv1_weight, Tensor::zeros::<f32>(vec![16, 8, 5, 5].into()));
    m.graph
        .inits
        .insert(reshape0_const, Tensor::new(vec![2].into(), vec![1i64, 256]));
    m.graph.inits.insert(
        reshape1_const0,
        Tensor::zeros::<f32>(vec![16, 4, 4, 10].into()),
    );
    m.graph.inits.insert(
        reshape1_const1,
        Tensor::new(vec![2].into(), vec![256i64, 10]),
    );

    let order = m.topo_sort_nodes();
    // println!(
    //     "{:#?}",
    //     order.iter().map(|&n| m.nodes[n].op).collect::<Vec<_>>()
    // );
    insta::assert_debug_snapshot!(order);
}
