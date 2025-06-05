use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::shape::{ShapeError, infer_shapes},
    model::Model,
    node::{Node, NodeId},
    op::{FusedElemwise, Op},
};

pub fn fuse_elemwise_ops(model: &mut Model) -> Result<(), ShapeError> {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();
    let value_parents = model.get_value_parents();

    let mut value_shapes = FxHashMap::default();
    infer_shapes(model, &mut FxHashMap::default(), &mut value_shapes)?;

    let mut list = vec![];
    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    for node_id in nodes {
        if visited.contains(&node_id) {
            continue;
        }

        let mut fusible_nodes = vec![];
        let mut last_node_id = None;
        let mut cur_node_id = node_id;
        loop {
            let node = &model.graph.nodes[cur_node_id];
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
                })
                && (last_node_id.is_none_or(|last_node_id| {
                    let last_node = &model.graph.nodes[last_node_id];
                    last_node.inputs.len() == 2
                        && last_node.outputs[0] == node.inputs[0]
                        && (node.inputs.len() == 1
                            || (node.inputs.len() == 2
                                && value_shapes[&node.inputs[1]]
                                    == value_shapes[&last_node.inputs[1]]))
                }));
            let end_of_chain = fusible
                && value_users
                    .get(&node.outputs[0])
                    .is_none_or(|users| users.len() != 1);
            if fusible {
                fusible_nodes.push(cur_node_id);
                last_node_id = Some(cur_node_id);
                if !end_of_chain {
                    cur_node_id = *value_users[&node.outputs[0]].iter().next().unwrap();
                    continue;
                }
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
                .map(|&id| model.graph.nodes[id]
                    .name
                    .as_deref()
                    .unwrap_or(model.graph.nodes[id].op.name()))
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }

    let count = list.len();

    for chain in list {
        let mut input_map = Vec::new();
        {
            let mut prev_node_id = None;
            for &node_id in &chain {
                let node = &model.graph.nodes[node_id];
                if let Some(prev) = prev_node_id {
                    input_map.extend(
                        node.inputs
                            .iter()
                            .filter(|i| !model.graph.nodes[prev].outputs.contains(i)),
                    );
                } else {
                    input_map.extend(node.inputs.iter());
                }
                prev_node_id = Some(node_id);
            }

            // Deduplicate values
            let mut present = FxHashSet::default();
            input_map.retain(|&id| present.insert(id));
        }
        let last_node = &model.graph.nodes[*chain.last().unwrap()];

        let fused_elemwise = Node::new(Op::FusedElemwise(FusedElemwise {
            input_map: input_map.clone(),
            chain: chain
                .iter()
                .map(|&id| {
                    let node = &model.graph.nodes[id];
                    (node.op.clone(), node.inputs.clone(), node.outputs.clone())
                })
                .collect(),
        }))
        .with_ins(input_map)
        .with_out(last_node.outputs[0]);
        model.graph.add_node(fused_elemwise);

        for &node_id in &chain {
            model.graph.nodes[node_id].deleted = true;
        }
    }

    log::info!("fuse_elemwise_ops({count}): {:?}", start.elapsed());

    Ok(())
}
