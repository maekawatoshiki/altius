use std::time::Instant;

use crate::{
    model::Model,
    node::Node,
    op::{Gemm, Op},
    value::ValueId,
};

pub fn fuse_transpose_matmul(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();
    let value_producer = model.get_value_parents();

    let mut list = vec![];
    let mut delete_list = vec![];

    enum MMInput {
        Transpose(ValueId),
        Other(ValueId),
    }

    impl MMInput {
        fn val(&self) -> ValueId {
            match self {
                MMInput::Transpose(v) => *v,
                MMInput::Other(v) => *v,
            }
        }
    }

    // TODO: WIP
    for node_id in nodes {
        let mm_id = node_id;
        let mm = &model.nodes[mm_id];
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }
        let mut lhs_transpose = false;
        let mut rhs_transpose = false;
        let mut lhs_input = MMInput::Other(mm.inputs[0]);
        let mut rhs_input = MMInput::Other(mm.inputs[1]);

        if let Some(&transpose_id) = value_producer.get(&mm.inputs[0]) {
            let transpose = &model.nodes[transpose_id];
            if matches!(transpose.op, Op::Transpose(ref t)
                if t.perm == [0, 1, 3, 2] || t.perm == [1, 0] || t.perm == [0, 2, 1])
            {
                lhs_transpose = true;
                lhs_input = MMInput::Transpose(transpose.inputs[0]);
                delete_list.push(transpose_id);
            }
        }
        if let Some(&transpose_id) = value_producer.get(&mm.inputs[1]) {
            let transpose = &model.nodes[transpose_id];
            if matches!(transpose.op, Op::Transpose(ref t)
                if t.perm == [0, 1, 3, 2] || t.perm == [1, 0] || t.perm == [0, 2, 1])
            {
                rhs_transpose = true;
                rhs_input = MMInput::Transpose(transpose.inputs[0]);
                delete_list.push(transpose_id);
            }
        }

        if lhs_transpose || rhs_transpose {
            list.push((lhs_input, rhs_input, mm.outputs[0]));
            delete_list.push(mm_id);
        }
    }

    let count = list.len();

    for (lhs_input, rhs_input, mm_output) in list {
        let gemm_out = model.values.new_val();
        let gemm = Node::new(Op::Gemm(Gemm {
            alpha: 1.0,
            beta: 0.0,
            trans_a: matches!(lhs_input, MMInput::Transpose(_)),
            trans_b: matches!(rhs_input, MMInput::Transpose(_)),
        }))
        .with_in(lhs_input.val())
        .with_in(rhs_input.val())
        .with_out(gemm_out);
        model.add_node(gemm);

        for user_id in &value_users[&mm_output] {
            let user = &mut model.nodes[*user_id];
            for i in &mut user.inputs {
                if *i == mm_output {
                    *i = gemm_out;
                }
            }
        }
    }

    for node in delete_list {
        model.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_transpose_matmul({count}): {:?}", start.elapsed());
}
