use crate::{dim::Dimensions, tensor::Tensor};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

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
    pub fn output_dims(&self) -> Option<&Dimensions> {
        match self {
            Self::Conv2d(n) => Some(&n.output_dims),
            Self::Add(n) => Some(&n.output_dims),
            Self::Relu(n) => Some(n.output_dims()),
            Self::MaxPool(n) => Some(&n.output_dims),
            Self::Reshape(n) => Some(&n.output_dims),
            Self::MatMul(n) => Some(&n.output_dims),
            Self::Tensor(n) => Some(&n.dims()),
            Self::Input(d) => Some(&d),
        }
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
        let relu =
            Relu::new(input_node.output_dims().unwrap().clone()).with_input_node(input_node_id);
        self.new(relu.into())
    }
}
