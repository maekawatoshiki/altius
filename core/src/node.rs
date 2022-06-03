use crate::{dim::Dimensions, tensor::Tensor};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

pub type Node2Id = Id<Node2>;
pub type Node2Arena = Arena<Node2>;

#[derive(Debug, Clone)]
pub struct Node2 {
    pub op: Op,
    pub attrs: Vec<Attr>,
    pub inputs: Vec<Node2Id>,
    pub outputs: Vec<Node2Id>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Conv2d,
    Add,
    ReLU,
    MaxPool,
    Reshape,
    MatMul,
    Const,
    Input,
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
