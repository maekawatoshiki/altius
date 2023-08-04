use rustc_hash::FxHashMap as HashMap;

use crate::{
    node::{Node, NodeArena, NodeId},
    tensor::Tensor,
    value::{ValueArena, ValueId},
};

#[derive(Default, Clone)]
pub struct Graph {
    pub nodes: NodeArena,
    pub values: ValueArena,
    pub inits: HashMap<ValueId, Tensor>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn add_node(&mut self, node: Node) -> NodeId {
        self.nodes.alloc(node)
    }
}
