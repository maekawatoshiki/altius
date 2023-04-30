use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::{model::Model, node::NodeId};

pub fn fuse_elemwise_ops(model: &Model) -> Vec<Vec<NodeId>> {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();
    let value_parents = model.get_value_parents();

    let mut list = vec![];
    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    for node_id in nodes {
        let mut fusible_nodes = vec![];
        let mut last_node_id = None;
        let mut cur_node_id = node_id;
        loop {
            let node = &model.nodes[cur_node_id];
            let fusible = node.op.is_elemwise()
                && node.outputs.len() == 1
                && node.inputs.iter().all(|id| {
                    // The input is either:
                    // - an initializer
                    // - a value from a previous node
                    // - a value of the first node in the chain
                    !value_parents.contains_key(id)
                        || (last_node_id.is_some() && Some(value_parents[id]) == last_node_id)
                        || last_node_id.is_none()
                });
            if fusible {
                fusible_nodes.push(cur_node_id);
                last_node_id = Some(cur_node_id);
                cur_node_id = *value_users[&node.outputs[0]].iter().next().unwrap();
                continue;
            }
            break;
        }

        if fusible_nodes.len() > 1 {
            visited.extend(fusible_nodes.iter());
            list.push(fusible_nodes);
        }
    }

    #[cfg(debug_assertions)]
    for nodes in list.iter() {
        log::debug!(
            "Fusible chain: {}",
            nodes
                .iter()
                .map(|&id| model.nodes[id]
                    .name
                    .as_ref()
                    .map(String::as_str)
                    .unwrap_or(model.nodes[id].op.name()))
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }

    log::info!("fuse_elemwise_ops(): {:?}", start.elapsed());

    list
}
