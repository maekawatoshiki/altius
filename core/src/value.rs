use id_arena::{Arena, Id};

pub type ValueId = Id<Value>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value;

#[derive(Default)]
pub struct ValueArena(Arena<Value>);

impl ValueArena {
    pub fn new_val(&mut self) -> ValueId {
        self.0.alloc(Value)
    }
}
