use std::time::Instant;

use crate::{
    model::Model,
    node::Node,
    op::{Gemm, Op, Squeeze},
    // tensor::Tensor,
};

pub fn fuse_matmul_add(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    // infer_shapes(model);

    let mut list = vec![];
    let mut delete_list = vec![];
    let mut count = 0;

    for node_id in nodes {
        let mm_id = node_id;
        let mm = &model.nodes[mm_id];
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }

        let mm_users = &value_users[&mm.outputs[0]];
        if mm_users.len() != 1 {
            continue;
        }
        let add_id = mm_users.iter().next().copied().unwrap();
        let add = &model.nodes[add_id];
        if !matches!(add.op, Op::Add) {
            continue;
        }
        let bias = if add.inputs[0] == mm.outputs[0] { 1 } else { 0 };

        list.push((
            mm.inputs[0],
            mm.inputs[1],
            mm.outputs[0],
            add.inputs[bias],
            add.outputs[0],
        ));
        delete_list.push(mm_id);
        delete_list.push(add_id);

        if count == 4 {
            break;
        }
        count += 1
    }

    let count = list.len();

    for (mm_lhs, mm_rhs, _mm_out, bias, add_out) in list {
        let squeeze_out = model.values.new_val();
        let squeeze = Node::new(Op::Squeeze(Squeeze { axes: vec![0] }))
            .with_in(mm_lhs)
            .with_out(squeeze_out);
        model.add_node(squeeze);

        let gemm_out = model.values.new_val();
        let gemm = Node::new(Op::Gemm(Gemm {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }))
        .with_in(squeeze_out)
        // .with_in(mm_lhs)
        .with_in(mm_rhs)
        .with_in(bias)
        .with_out(gemm_out);
        model.add_node(gemm);

        for user_id in &value_users[&add_out] {
            let user = &mut model.nodes[*user_id];
            let idx = user.inputs.iter().position(|&i| i == add_out).unwrap();
            user.inputs[idx] = gemm_out
        }
    }

    for node in delete_list {
        model.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_matmul_add({count}): {:?}", start.elapsed());
}
