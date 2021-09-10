use crate::node::{NodeArena, NodeId};

pub struct Model {
    nodes: NodeArena,
    input_node: Option<NodeId>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            nodes: NodeArena::new(),
            input_node: None,
        }
    }
}
