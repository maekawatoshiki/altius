use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use crate::{
    model::Model,
    node::{NodeBuilder, NodeId},
};

pub fn infer_shapes(model: &mut Model) {
    let mut worklist = VecDeque::new();
    create_worklist(model, model.output_node.unwrap(), &mut worklist);
    let mut visited = FxHashSet::default();
    for node_id in worklist {
        if !visited.insert(node_id) {
            continue;
        }
        let node = &model.arena()[node_id];
        let mut input_dims = vec![];
        for input in node.input() {
            input_dims.push(model.arena()[input.unwrap()].output_dims().clone());
        }
        let node = &mut model.arena_mut()[node_id];
        for (i, dims) in input_dims.into_iter().enumerate() {
            node.input_dims_mut()[i] = dims;
        }
        node.compute_output_dims();
    }
}

fn create_worklist(
    model: &Model,
    node_id: NodeId,
    worklist: &mut VecDeque<NodeId>, // shapes: &mut FxHashMap<String, Dimensions>,
) {
    let node = &model.arena()[node_id];
    for input in node.input() {
        create_worklist(model, input.unwrap(), worklist);
    }
    for input in node.input() {
        worklist.push_back(input.unwrap())
    }
    worklist.push_back(node_id);
}
