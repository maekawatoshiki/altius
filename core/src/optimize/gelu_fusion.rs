use std::time::Instant;

use crate::{model::Model, node::Node, op::Op};

pub fn fuse_gelu(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    let mut list = vec![];
    let mut delete_list = vec![];

    for node_id in nodes {
        let div_id = node_id;
        let div = &model.nodes[div_id];
        if !matches!(div.op, Op::Div) {
            continue;
        }
        let approx_sqrt_two = 1.4142099618911743f32;
        if model.inits.get(&div.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || !rhs.allclose_f32(&[approx_sqrt_two])
        }) {
            continue;
        }

        let erf_id = value_users[&div.outputs[0]].iter().next().copied().unwrap();
        let erf = &model.nodes[erf_id];
        if !matches!(erf.op, Op::Erf) {
            continue;
        }

        let add_id = value_users[&erf.outputs[0]].iter().next().copied().unwrap();
        let add = &model.nodes[add_id];
        if !matches!(add.op, Op::Add) {
            continue;
        }
        let is_erf_add_lhs = add.inputs[0] == erf.outputs[0];
        if model
            .inits
            .get(&add.inputs[is_erf_add_lhs as usize])
            .map_or(true, |one| {
                !one.elem_ty().is_f32() || one.data::<f32>()[0] != 1.
            })
        {
            continue;
        }

        let mul1_id = value_users[&add.outputs[0]].iter().next().copied().unwrap();
        let mul1 = &model.nodes[mul1_id];
        if !matches!(mul1.op, Op::Mul) {
            continue;
        }
        let is_add_mul1_lhs = mul1.inputs[0] == add.outputs[0];
        if mul1.inputs[is_add_mul1_lhs as usize] != div.inputs[0] {
            continue;
        }

        let mul2_id = value_users[&mul1.outputs[0]]
            .iter()
            .next()
            .copied()
            .unwrap();
        let mul2 = &model.nodes[mul2_id];
        if !matches!(mul2.op, Op::Mul) {
            continue;
        }
        let is_mul1_mul2_lhs = mul2.inputs[0] == mul1.outputs[0];
        if model
            .inits
            .get(&mul2.inputs[is_mul1_mul2_lhs as usize])
            .map_or(true, |half| {
                !half.elem_ty().is_f32() || half.data::<f32>()[0] != 0.5
            })
        {
            continue;
        }

        // Gelu Detected!

        list.push((div.inputs[0], mul2.outputs[0]));
        delete_list.push(div_id);
        delete_list.push(erf_id);
        delete_list.push(add_id);
        delete_list.push(mul1_id);
        delete_list.push(mul2_id);
    }

    let count = list.len();

    for (start, end) in list {
        let gelu_out = model.values.new_val();
        let gelu = Node::new(Op::Gelu).with_in(start).with_out(gelu_out);
        let _gelu_id = model.add_node(gelu);

        for user_id in &value_users[&end] {
            let user = &mut model.nodes[*user_id];
            let idx = user.inputs.iter().position(|&i| i == end).unwrap();
            user.inputs[idx] = gelu_out
        }
    }

    for node in delete_list {
        model.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_gelu({count}): {:?}", start.elapsed());
}
