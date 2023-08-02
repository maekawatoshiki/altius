use rustc_hash::FxHashMap as HashMap;

use crate::{
    node::NodeArena,
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
