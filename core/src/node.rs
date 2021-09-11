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
}

#[derive(Default)]
pub struct Conv2d {
    pub input_dims: Dimensions,
    pub weight_dims: Dimensions,
    pub weight: Tensor,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Add {
    pub input_a_dims: Dimensions,
    pub input_b_dims: Dimensions,
    pub input_b: Tensor,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Relu {
    pub input_dims: Dimensions,
    pub output_dims: Dimensions,
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
    pub data: Option<Tensor>,
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

pub trait NodeBuilder {
    fn arena(&self) -> &NodeArena;
    fn arena_mut(&mut self) -> &mut NodeArena;
    fn new(&mut self, node: Node) -> NodeId {
        self.arena_mut().alloc(node)
    }
}
