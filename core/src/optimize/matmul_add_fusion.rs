use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::{
    model::Model,
    node::Node,
    op::{infer_shapes, Gemm, Op, Squeeze, Unsqueeze},
    tensor::Tensor,
    // tensor::Tensor,
};

pub fn fuse_matmul_add(model: &mut Model) {
    // TODO
    log::warn!("fuse_matmul_add: This optimization pass is not implemented for general cases!!!");
    log::warn!("fuse_matmul_add: In most cases, it does not perform a correct optimization!!!");

    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    let seq = model.topo_sort_nodes();
    let mut shapes = FxHashMap::default();
    infer_shapes(model, &seq, &mut shapes).unwrap();

    let mut list = vec![];
    let mut delete_list = vec![];
    let mut count = 0;

    for node_id in nodes {
        let mm_id = node_id;
        let mm = &model.nodes[mm_id];
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }
        let a = &shapes[&mm_id];
        if a.1[0].dims.len() == 3 {
            // println!(">>> {:?}", a);
        } else {
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
            // break;
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

        let xx = model.values.new_val();
        model
            .inits
            .insert(xx, Tensor::new(vec![1].into(), vec![0i64]));
        let unsqueeze_out = model.values.new_val();
        let unsqueeze = Node::new(Op::Unsqueeze(Unsqueeze { axes: vec![] }))
            .with_in(gemm_out)
            .with_in(xx)
            .with_out(unsqueeze_out);
        model.add_node(unsqueeze);

        for user_id in &value_users[&add_out] {
            let user = &mut model.nodes[*user_id];
            let idx = user.inputs.iter().position(|&i| i == add_out).unwrap();
            user.inputs[idx] = unsqueeze_out;
        }
    }

    for node in delete_list {
        model.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_matmul_add({count}): {:?}", start.elapsed());
}
