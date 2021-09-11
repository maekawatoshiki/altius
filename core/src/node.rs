use crate::{dim::Dimensions, tensor::Tensor};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

pub enum Node {
    Conv2d(Conv2d),
}

#[derive(Default)]
pub struct Conv2d {
    pub input_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub output_dims: Dimensions,
    pub next_node: Option<NodeId>,
}

impl From<Conv2d> for Node {
    fn from(n: Conv2d) -> Node {
        Node::Conv2d(n)
    }
}

pub trait NodeBuilder {
    fn arena(&self) -> &NodeArena;
    fn arena_mut(&mut self) -> &mut NodeArena;
    fn new(&mut self, node: Node) -> NodeId {
        self.arena_mut().alloc(node)
    }
}
