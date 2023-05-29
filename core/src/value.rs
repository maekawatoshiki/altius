use std::ops::{Index, IndexMut};

use id_arena::{Arena, Id};

use crate::tensor::{TypedFixedShape, TypedSymbolicShape};

pub type ValueId = Id<Value>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value {
    pub name: Option<String>,
    pub shape: Option<TypedFixedShape>, // TODO: Support dynamic shape.
    pub symshape: Option<TypedSymbolicShape>, // This will replace `shape` in the future.
}

#[derive(Debug, Default, Clone)]
pub struct ValueArena(Arena<Value>);

impl ValueArena {
    pub fn new_val(&mut self) -> ValueId {
        self.0.alloc(Value {
            name: None,
            shape: None,
            symshape: None,
        })
    }

    pub fn new_val_named(&mut self, name: impl Into<String>) -> ValueId {
        self.0.alloc(Value {
            name: Some(name.into()),
            shape: None,
            symshape: None,
        })
    }

    pub fn new_val_named_and_shaped(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<TypedFixedShape>,
    ) -> ValueId {
        self.0.alloc(Value {
            name: Some(name.into()),
            shape: Some(shape.into()),
            symshape: None,
        })
    }

    pub fn new_val_named_and_symshaped(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<TypedSymbolicShape>,
    ) -> ValueId {
        self.0.alloc(Value {
            name: Some(name.into()),
            shape: None,
            symshape: Some(shape.into()),
        })
    }

    pub fn inner(&self) -> &Arena<Value> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut Arena<Value> {
        &mut self.0
    }
}

impl Index<ValueId> for ValueArena {
    type Output = Value;

    fn index(&self, index: ValueId) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<ValueId> for ValueArena {
    fn index_mut(&mut self, index: ValueId) -> &mut Self::Output {
        &mut self.0[index]
    }
}
