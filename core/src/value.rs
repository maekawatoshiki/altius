use id_arena::{Arena, Id};

pub type ValueId = Id<Value>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value(pub Option<String>);

#[derive(Default, Clone)]
pub struct ValueArena(Arena<Value>);

impl ValueArena {
    pub fn new_val(&mut self) -> ValueId {
        self.0.alloc(Value(None))
    }

    pub fn new_val_named(&mut self, name: impl Into<String>) -> ValueId {
        self.0.alloc(Value(Some(name.into())))
    }

    pub fn inner(&self) -> &Arena<Value> {
        &self.0
    }
}
