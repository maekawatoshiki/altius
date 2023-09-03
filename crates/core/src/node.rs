use crate::{op::Op, value::ValueId};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub name: Option<String>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub deleted: bool,
}

impl Node {
    pub fn new(op: Op) -> Self {
        Self {
            op,
            name: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            deleted: false,
        }
    }

    pub fn with_name(mut self, name: impl Into<Option<String>>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_in(mut self, id: ValueId) -> Self {
        self.inputs.push(id);
        self
    }

    pub fn with_ins(mut self, mut ids: Vec<ValueId>) -> Self {
        self.inputs.append(&mut ids);
        self
    }

    pub fn with_out(mut self, id: ValueId) -> Self {
        self.outputs.push(id);
        self
    }

    pub fn with_outs(mut self, mut ids: Vec<ValueId>) -> Self {
        self.outputs.append(&mut ids);
        self
    }

    pub fn alloc(self, arena: &mut NodeArena) -> NodeId {
        arena.alloc(self)
    }
}
