use std::time::Instant;

use crate::{model::Model, node::Node, op::Op};

pub fn fuse_transpose_matmul(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();
    let value_producer = model.get_value_parents();

    let mut list = vec![];
    let mut delete_list = vec![];

    // TODO: WIP
    for node_id in nodes {
        let mm_id = node_id;
        let mm = &model.nodes[mm_id];
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }
        if let Some(&lhs) = value_producer.get(&mm.inputs[0]) {
            let transpose = &model.nodes[lhs];
            if matches!(transpose.op, Op::Transpose(_)) {
                list.push((transpose.inputs[0], mm.outputs[0]));
                delete_list.push(lhs);
                delete_list.push(mm_id);
            }
        }
        // let approx_sqrt_two = 1.4142099618911743f32;
        // if model.inits.get(&div.inputs[1]).map_or(true, |rhs| {
        //     !rhs.elem_ty().is_f32() || !allclose(rhs.data::<f32>(), &[approx_sqrt_two])
        // }) {
        //     continue;
        // }

        // Gelu Detected!

        // list.push((div.inputs[0], mul2.outputs[0]));
        // delete_list.push(div_id);
        // delete_list.push(erf_id);
        // delete_list.push(add_id);
        // delete_list.push(mul1_id);
        // delete_list.push(mul2_id);
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

    log::info!("fuse_transpose_matmul({count}): {:?}", start.elapsed());
}
