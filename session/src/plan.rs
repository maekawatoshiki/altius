use altius_core::{model::Model, node::NodeId, value::ValueId};
use rustc_hash::FxHashMap;

/// Represents a node to execute and values to be freed after the execution of the node.
#[derive(Debug)]
pub struct NodeExecutionPlan {
    /// The node to execute.
    pub node_id: NodeId,

    /// Values to be freed after the execution of the node.
    pub free_vals: Vec<ValueId>,
}

pub fn create_execution_plan(model: &Model) -> Vec<NodeExecutionPlan> {
    let sorted_nodes = model.topo_sort_nodes();
    let node_order: FxHashMap<NodeId, usize> = sorted_nodes
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut new_sorted_nodes = vec![];
    let mut node_to_free_vals = FxHashMap::default();
    let value_users = model.get_value_users();

    for node_id in sorted_nodes {
        let node = &model.nodes[node_id];
        let mut plan = NodeExecutionPlan {
            node_id,
            free_vals: Vec::new(),
        };

        for &output_id in &node.outputs {
            if !value_users.contains_key(&output_id) {
                continue;
            }

            let users = &value_users[&output_id];
            let last_user = users
                .iter()
                .map(|id| (node_order[id], id))
                .max_by(|x, y| x.0.cmp(&y.0))
                .unwrap()
                .1;
            node_to_free_vals
                .entry(last_user)
                .or_insert_with(Vec::new)
                .push(output_id)
        }

        if let Some(mut vals) = node_to_free_vals.remove(&node_id) {
            plan.free_vals.append(&mut vals);
        }

        new_sorted_nodes.push(plan);
    }

    new_sorted_nodes
}
