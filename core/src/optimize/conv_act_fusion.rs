use std::time::Instant;

use crate::{
    model::Model,
    op::{FusedActivation, Op},
};

pub fn fuse_conv_act(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    let mut list = vec![];
    let mut delete_list = vec![];

    for node_id in nodes {
        let conv_id = node_id;
        let conv = &model.nodes[conv_id];
        if !matches!(conv.op, Op::Conv2d(_)) {
            continue;
        }
        if value_users[&conv.outputs[0]].len() != 1 {
            continue;
        }

        let act_id = value_users[&conv.outputs[0]]
            .iter()
            .next()
            .copied()
            .unwrap();
        let act = &model.nodes[act_id];
        let fused_act = match act.op {
            Op::ReLU => FusedActivation::Relu,
            Op::HardSigmoid(ref h) => FusedActivation::HardSigmoid(*h),
            _ => continue,
        };

        // Conv+Activation Detected!

        list.push((fused_act, conv_id, conv.outputs[0], act.outputs[0]));
        delete_list.push(act_id);
    }

    let count = list.len();

    for (fused_act, conv_id, conv_out, act) in list {
        if let Op::Conv2d(ref mut c) = &mut model.nodes[conv_id].op {
            c.activation = Some(fused_act);
        }

        for user_id in &value_users[&act] {
            let user = &mut model.nodes[*user_id];
            let idx = user.inputs.iter().position(|&i| i == act).unwrap();
            user.inputs[idx] = conv_out
        }
    }

    for node in delete_list {
        model.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_conv_act({count}): {:?}", start.elapsed());
}
