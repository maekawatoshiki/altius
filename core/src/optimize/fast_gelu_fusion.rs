use std::time::Instant;

use crate::{model::Model, node::Node, op::Op};

pub fn fuse_fast_gelu(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();
    let value_parents = model.get_value_parents();

    let mut list = vec![];
    let mut delete_list = vec![];

    for node_id in nodes {
        let node = &model.nodes[node_id];
        if node.outputs.len() != 1 {
            continue;
        }

        let x = node.outputs[0];

        let Some(nodes) = value_users.get(&x) else {
            continue;
        };

        let Some(&pow_id) = nodes.iter().find(|&&node_id| {
            let node = &model.nodes[node_id];
            matches!(node.op, Op::Pow)
        }) else {
            continue;
        };
        let pow = &model.nodes[pow_id];
        if model.inits.get(&pow.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || rhs.data::<f32>()[0] != 3.
        }) {
            continue;
        }
        if value_users[&pow.outputs[0]].len() != 1 {
            continue;
        }

        let mul1_id = value_users[&pow.outputs[0]].iter().next().copied().unwrap();
        let mul1 = &model.nodes[mul1_id];
        if !matches!(mul1.op, Op::Mul) {
            continue;
        }
        if model.inits.get(&mul1.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || !rhs.allclose_f32(&[0.044714998453855515])
        }) {
            continue;
        }
        if value_users[&mul1.outputs[0]].len() != 1 {
            continue;
        }

        let add1_id = value_users[&mul1.outputs[0]]
            .iter()
            .next()
            .copied()
            .unwrap();
        let add = &model.nodes[add1_id];
        if !matches!(add.op, Op::Add) {
            continue;
        }
        if add.inputs[0] != x {
            continue;
        }
        if value_users[&add.outputs[0]].len() != 1 {
            continue;
        }

        let mul2_id = value_users[&add.outputs[0]].iter().next().copied().unwrap();
        let mul2 = &model.nodes[mul2_id];
        if !matches!(mul2.op, Op::Mul) {
            continue;
        }
        if model.inits.get(&mul2.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || !rhs.allclose_f32(&[0.7978845834732056])
        }) {
            continue;
        }
        if value_users[&mul2.outputs[0]].len() != 1 {
            continue;
        }

        let tanh_id = value_users[&mul2.outputs[0]]
            .iter()
            .next()
            .copied()
            .unwrap();
        let tanh = &model.nodes[tanh_id];
        if !matches!(tanh.op, Op::Tanh) {
            continue;
        }
        if value_users[&tanh.outputs[0]].len() != 1 {
            continue;
        }

        let add2_id = value_users[&tanh.outputs[0]]
            .iter()
            .next()
            .copied()
            .unwrap();
        let add = &model.nodes[add2_id];
        if !matches!(add.op, Op::Add) {
            continue;
        }
        if model.inits.get(&add.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || rhs.data::<f32>()[0] != 1.
        }) {
            continue;
        }
        if value_users[&add.outputs[0]].len() != 1 {
            continue;
        }

        let mul3_id = value_users[&add.outputs[0]].iter().next().copied().unwrap();
        let mul3 = &model.nodes[mul3_id];
        assert_eq!(mul3.inputs[1], add.outputs[0]);
        if !matches!(mul3.op, Op::Mul) {
            continue;
        }

        let mul4_id = value_parents[&mul3.inputs[0]];
        let mul4 = &model.nodes[mul4_id];
        if !matches!(mul4.op, Op::Mul) {
            continue;
        }
        if mul4.inputs[0] != x {
            continue;
        }
        if model.inits.get(&mul4.inputs[1]).map_or(true, |rhs| {
            !rhs.elem_ty().is_f32() || rhs.data::<f32>()[0] != 0.5
        }) {
            continue;
        }
        if value_users[&mul4.outputs[0]].len() != 1 {
            continue;
        }

        // Fast Gelu Detected!

        list.push((x, mul3.outputs[0]));
        delete_list.push(pow_id);
        delete_list.push(mul1_id);
        delete_list.push(add1_id);
        delete_list.push(mul2_id);
        delete_list.push(tanh_id);
        delete_list.push(add2_id);
        delete_list.push(mul3_id);
        delete_list.push(mul4_id);
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

    log::info!("fuse_fast_gelu({count}): {:?}", start.elapsed());
}
