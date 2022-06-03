use crate::{
    dim::Dimensions,
    tensor::{Tensor, Tensor2},
    value::ValueId,
};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

pub type Node2Id = Id<Node2>;
pub type Node2Arena = Arena<Node2>;

#[derive(Debug, Clone)]
pub struct Node2 {
    pub op: Op,
    pub attrs: Vec<Attr>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Conv2d,
    Add,
    ReLU,
    MaxPool,
    Reshape,
    MatMul,
}

#[derive(Debug, Clone)]
pub enum Attr {
    Shape(Dimensions),
}

pub enum Node {
    Conv2d(Conv2d),
    Add(Add),
    Relu(Relu),
    MaxPool(MaxPool),
    Reshape(Reshape),
    MatMul(MatMul),
    Tensor(Tensor),
    Input(Dimensions),
}

impl Node2 {
    pub const CONV2D_ATTR_KERNEL: usize = 0;
    pub const CONV2D_ATTR_STRIDE: usize = 1;
    pub const CONV2D_ATTR_PADDING: usize = 2;
    pub const CONV2D_IN: usize = 0;
    pub const CONV2D_WEIGHT: usize = 1;
    pub const CONV2D_OUT: usize = 0;

    pub const ADD_IN_A: usize = 0;
    pub const ADD_IN_B: usize = 1;
    pub const ADD_OUT: usize = 0;

    pub const RELU_IN: usize = 0;
    pub const RELU_OUT: usize = 0;

    pub const MAXPOOL_ATTR_KERNEL: usize = 0;
    pub const MAXPOOL_ATTR_STRIDE: usize = 1;
    pub const MAXPOOL_IN: usize = 0;
    pub const MAXPOOL_OUT: usize = 0;

    pub const RESHAPE_IN: usize = 0;
    pub const RESHAPE_SHAPE: usize = 1;
    pub const RESHAPE_OUT: usize = 0;

    pub const MATMUL_IN_A: usize = 0;
    pub const MATMUL_IN_B: usize = 1;
    pub const MATMUL_OUT: usize = 0;

    pub fn new(op: Op) -> Self {
        Self {
            op,
            attrs: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_attr(mut self, attr: Attr) -> Self {
        self.attrs.push(attr);
        self
    }

    pub fn with_in(mut self, id: ValueId) -> Self {
        self.inputs.push(id);
        self
    }

    pub fn with_out(mut self, id: ValueId) -> Self {
        self.outputs.push(id);
        self
    }

    pub fn alloc(self, arena: &mut Node2Arena) -> Node2Id {
        let id = arena.alloc(self);
        id
    }

    pub fn compute_output_shapes(&self, inputs: &[Tensor2]) -> Vec<Dimensions> {
        let mut shapes = vec![];
        match self.op {
            Op::Conv2d => {
                let Attr::Shape(kernel) = &self.attrs[Self::CONV2D_ATTR_KERNEL];
                let Attr::Shape(stride) = &self.attrs[Self::CONV2D_ATTR_STRIDE];
                let Attr::Shape(padding) = &self.attrs[Self::CONV2D_ATTR_PADDING];
                let input = inputs[Self::CONV2D_IN].dims();
                let weight = inputs[Self::CONV2D_WEIGHT].dims();

                let h_in = input.as_slice()[2];
                let w_in = input.as_slice()[3];
                let output_shape = vec![
                    input.as_slice()[0],
                    weight.as_slice()[0],
                    (h_in + 2 * padding.as_slice()[0] - 1 * (kernel.as_slice()[0] - 1) - 1)
                        / stride.as_slice()[0]
                        + 1,
                    (w_in + 2 * padding.as_slice()[1] - 1 * (kernel.as_slice()[1] - 1) - 1)
                        / stride.as_slice()[1]
                        + 1,
                ];
                shapes.push(output_shape.into());
            }
            Op::Add => {
                let in_a = inputs[Self::ADD_IN_A].dims();
                let in_b = inputs[Self::ADD_IN_B].dims();
                assert!(
                    in_a == in_b || {
                        in_a.len() == 4
                            && in_b.len() == 3
                            && in_a.as_slice()[1] == in_b.as_slice()[0]
                            && in_b.as_slice()[1] == 1
                            && in_b.as_slice()[2] == 1
                    }
                ); // TODO: Support broadcasting.
                shapes.push(in_a.clone());
            }
            Op::MaxPool => {
                let Attr::Shape(kernel) = &self.attrs[Self::MAXPOOL_ATTR_KERNEL];
                let Attr::Shape(stride) = &self.attrs[Self::MAXPOOL_ATTR_STRIDE];
                let input = &inputs[Self::MAXPOOL_IN].dims();

                let h_in = input.as_slice()[2];
                let w_in = input.as_slice()[3];
                let output_shape = vec![
                    input.as_slice()[0],
                    input.as_slice()[1],
                    (h_in + 2 * 0 - 1 * (kernel.as_slice()[0] - 1) - 1) / stride.as_slice()[0] + 1,
                    (w_in + 2 * 0 - 1 * (kernel.as_slice()[1] - 1) - 1) / stride.as_slice()[1] + 1,
                ];
                shapes.push(output_shape.into());
            }
            Op::Reshape => {
                let shape = inputs[Self::RESHAPE_SHAPE]
                    .data()
                    .as_i64()
                    .unwrap()
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>();
                shapes.push(shape.into());
            }
            Op::MatMul => {
                let in_a = &inputs[Self::MATMUL_IN_A].dims();
                let in_b = &inputs[Self::MATMUL_IN_B].dims();
                assert_eq!(in_a.as_slice()[1], in_b.as_slice()[0]);
                shapes.push(vec![in_a.as_slice()[0], in_b.as_slice()[1]].into());
            }
            Op::ReLU => {
                let input = inputs[Self::RELU_IN].dims();
                shapes.push(input.clone());
            }
        }
        shapes
    }
}

impl From<Vec<usize>> for Attr {
    fn from(v: Vec<usize>) -> Self {
        Attr::Shape(Dimensions(v))
    }
}

#[derive(Default)]
pub struct Conv2d {
    pub input_dims: Dimensions,
    pub weight_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub padding: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
    pub weight_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Add {
    pub input_a_dims: Dimensions,
    pub input_b_dims: Dimensions,
    pub input_b: Tensor,
    pub output_dims: Dimensions,
    pub input_a_node: Option<NodeId>,
    pub input_b_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Relu {
    pub input_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct MaxPool {
    pub input_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Reshape {
    pub input_dims: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct MatMul {
    pub input_a_dims: Dimensions,
    pub input_b_dims: Dimensions,
    pub output_dims: Dimensions,
    pub input_a_node: Option<NodeId>,
    pub input_b_node: Option<NodeId>,
}

impl Node {
    pub fn output_dims(&self) -> &Dimensions {
        match self {
            Self::Conv2d(n) => &n.output_dims,
            Self::Add(n) => &n.output_dims,
            Self::Relu(n) => n.output_dims(),
            Self::MaxPool(n) => &n.output_dims,
            Self::Reshape(n) => &n.output_dims,
            Self::MatMul(n) => &n.output_dims,
            Self::Tensor(n) => &n.dims(),
            Self::Input(d) => &d,
        }
    }
}

impl Conv2d {
    pub fn new(input_dims: Dimensions, kernel: Dimensions) -> Self {
        Self {
            input_dims,
            kernel,
            ..Conv2d::default()
        }
    }

    pub fn with_strides(mut self, strides: Dimensions) -> Self {
        self.stride = strides;
        self
    }

    pub fn with_padding(mut self, padding: Dimensions) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_input_node(mut self, input_node: NodeId) -> Self {
        self.input_node = Some(input_node);
        self
    }

    pub fn with_weight_node(mut self, weight_node: NodeId, weight_dims: Dimensions) -> Self {
        self.weight_node = Some(weight_node);
        self.weight_dims = weight_dims;
        self
    }

    pub fn with_output_dims(mut self, output_dims: Dimensions) -> Self {
        self.output_dims = output_dims;
        self
    }
}

impl Relu {
    pub fn new(input_dims: Dimensions) -> Self {
        Self {
            input_dims,
            ..Relu::default()
        }
    }

    pub fn with_input_node(mut self, input_node: NodeId) -> Self {
        self.input_node = Some(input_node);
        self
    }

    pub fn output_dims(&self) -> &Dimensions {
        &self.input_dims
    }
}

impl From<Conv2d> for Node {
    fn from(n: Conv2d) -> Node {
        Node::Conv2d(n)
    }
}

impl From<Add> for Node {
    fn from(n: Add) -> Node {
        Node::Add(n)
    }
}

impl From<Relu> for Node {
    fn from(n: Relu) -> Node {
        Node::Relu(n)
    }
}

impl From<MaxPool> for Node {
    fn from(n: MaxPool) -> Node {
        Node::MaxPool(n)
    }
}

impl From<Reshape> for Node {
    fn from(n: Reshape) -> Node {
        Node::Reshape(n)
    }
}

impl From<MatMul> for Node {
    fn from(n: MatMul) -> Node {
        Node::MatMul(n)
    }
}

impl From<Tensor> for Node {
    fn from(n: Tensor) -> Node {
        Node::Tensor(n)
    }
}

pub trait NodeBuilder {
    fn arena(&self) -> &NodeArena;
    fn arena_mut(&mut self) -> &mut NodeArena;

    fn new(&mut self, node: Node) -> NodeId {
        self.arena_mut().alloc(node)
    }

    fn new_relu(&mut self, input_node_id: NodeId) -> NodeId {
        let input_node = &self.arena()[input_node_id];
        let relu = Relu::new(input_node.output_dims().clone()).with_input_node(input_node_id);
        self.new(relu.into())
    }

    fn new_conv2d(
        &mut self,
        input_node_id: NodeId,
        weight_node_id: NodeId,
        kernel: Dimensions,
        strides: Dimensions,
        padding: Dimensions,
    ) -> NodeId {
        let input_node = &self.arena()[input_node_id];
        let weight_node = &self.arena()[weight_node_id];
        let output_dims = {
            let h_in = input_node.output_dims().as_slice()[2];
            let w_in = input_node.output_dims().as_slice()[3];
            vec![
                input_node.output_dims().as_slice()[0],
                weight_node.output_dims().as_slice()[0],
                (h_in + 2 * padding.as_slice()[0] - 1 * (kernel.as_slice()[0] - 1) - 1)
                    / strides.as_slice()[0]
                    + 1,
                (w_in + 2 * padding.as_slice()[1] - 1 * (kernel.as_slice()[1] - 1) - 1)
                    / strides.as_slice()[1]
                    + 1,
            ]
        };
        let conv2d = Conv2d::new(input_node.output_dims().clone(), kernel)
            .with_input_node(input_node_id)
            .with_weight_node(weight_node_id, weight_node.output_dims().clone())
            .with_strides(strides)
            .with_padding(padding)
            .with_output_dims(output_dims.into());
        self.new(conv2d.into())
    }
}
