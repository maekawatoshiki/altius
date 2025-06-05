use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    model::Model,
    node::{Node, NodeId},
    op::Op,
    value::ValueId,
};

fn extract_gelu(
    model: &Model,
    value_users: &FxHashMap<ValueId, FxHashSet<NodeId>>,
    root_id: NodeId,
) -> Option<(ValueId, ValueId, Vec<NodeId>)> {
    let div = &model.graph.nodes[root_id];
    let approx_sqrt_two = 1.4142099618911743f32;
    if div.op != Op::Div
        || model
            .graph
            .inits
            .get(&div.inputs[1])
            .is_none_or(|rhs| !rhs.allclose(&[approx_sqrt_two]))
    {
        return None;
    }

    let erf_id = value_users[&div.outputs[0]].iter().next().copied()?;
    let erf = &model.graph.nodes[erf_id];
    if erf.op != Op::Erf {
        return None;
    }

    let add_id = value_users[&erf.outputs[0]].iter().next().copied()?;
    let add = &model.graph.nodes[add_id];
    if add.op != Op::Add {
        return None;
    }
    let is_erf_add_lhs = add.inputs[0] == erf.outputs[0];
    if model
        .graph
        .inits
        .get(&add.inputs[is_erf_add_lhs as usize])
        .is_none_or(|one| !one.elem_ty().is_f32() || one.data::<f32>()[0] != 1.)
    {
        return None;
    }

    let mul1_id = value_users[&add.outputs[0]].iter().next().copied()?;
    let mul1 = &model.graph.nodes[mul1_id];
    if mul1.op != Op::Mul {
        return None;
    }
    let is_add_mul1_lhs = mul1.inputs[0] == add.outputs[0];
    if mul1.inputs[is_add_mul1_lhs as usize] != div.inputs[0] {
        return None;
    }

    let mul2_id = value_users[&mul1.outputs[0]].iter().next().copied()?;
    let mul2 = &model.graph.nodes[mul2_id];
    if mul2.op != Op::Mul {
        return None;
    }
    let is_mul1_mul2_lhs = mul2.inputs[0] == mul1.outputs[0];
    if model
        .graph
        .inits
        .get(&mul2.inputs[is_mul1_mul2_lhs as usize])
        .is_none_or(|half| !half.elem_ty().is_f32() || half.data::<f32>()[0] != 0.5)
    {
        return None;
    }

    // Gelu Detected!

    Some((
        div.inputs[0],
        mul2.outputs[0],
        vec![root_id, erf_id, add_id, mul1_id, mul2_id],
    ))
}

pub fn fuse_gelu(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    let mut subgraphs = vec![];
    let mut unnecessary_nodes = vec![];

    for node_id in nodes {
        if let Some((start, end, nodes)) = extract_gelu(model, &value_users, node_id) {
            subgraphs.push((start, end));
            unnecessary_nodes.extend(nodes);
        }
    }

    let count = subgraphs.len();

    for (start, end) in subgraphs {
        let gelu_out = model.graph.values.new_val();
        let gelu = Node::new(Op::Gelu).with_in(start).with_out(gelu_out);
        model.graph.add_node(gelu);

        let Some(users) = value_users.get(&end) else {
            for output in &mut model.graph.outputs {
                if *output == end {
                    *output = gelu_out;
                }
            }
            continue;
        };
        for &user_id in users {
            let user = &mut model.graph.nodes[user_id];
            for input in &mut user.inputs {
                if *input == end {
                    *input = gelu_out;
                }
            }
        }
    }

    for node in unnecessary_nodes {
        model.graph.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_gelu({count}): {:?}", start.elapsed());
}
