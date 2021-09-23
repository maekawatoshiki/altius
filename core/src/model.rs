use crate::node::{NodeArena, NodeBuilder, NodeId};

pub struct Model {
    nodes: NodeArena,
    pub input_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
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
