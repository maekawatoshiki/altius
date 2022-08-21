use id_arena::{Arena, Id};

use crate::dim::Dimensions;

pub type ValueId = Id<Value>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value(pub Option<String>, pub Option<Dimensions>); // TODO: Support dynamic shape.

#[derive(Debug, Default, Clone)]
pub struct ValueArena(Arena<Value>);

impl ValueArena {
    pub fn new_val(&mut self) -> ValueId {
        self.0.alloc(Value(None, None))
    }

    pub fn new_val_named(&mut self, name: impl Into<String>) -> ValueId {
        self.0.alloc(Value(Some(name.into()), None))
    }

    pub fn new_val_named_and_shaped(
        &mut self,
        name: impl Into<String>,
        dims: impl Into<Dimensions>,
    ) -> ValueId {
        self.0.alloc(Value(Some(name.into()), Some(dims.into())))
    }

    pub fn inner(&self) -> &Arena<Value> {
        &self.0
    }
}
