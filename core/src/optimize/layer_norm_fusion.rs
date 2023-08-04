use std::time::Instant;

use crate::{
    model::Model,
    node::Node,
    op::{LayerNormalization, Op},
};

// From ONNX Runtime:
// +---------------------+
// |                     |
// |                     v
// X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                       |                                               ^
//                       |                                               |
//                       +-----------------------------------------------+

pub fn fuse_layer_norm(model: &mut Model) {
    let start = Instant::now();
    let nodes = model.topo_sort_nodes();
    let value_users = model.get_value_users();

    let mut list = vec![];
    let mut delete_list = vec![];

    for node_id in nodes {
        let mean_id = node_id;
        let mean = &model.graph.nodes[mean_id];
        if !matches!(mean.op, Op::ReduceMean(_)) {
            continue;
        }

        let Some(users) = value_users.get(&mean.outputs[0]) else {
            continue;
        };
        let sub_id = users.iter().next().copied().unwrap();
        let sub = &model.graph.nodes[sub_id];
        if !matches!(sub.op, Op::Sub) {
            continue;
        }

        let Some(users) = value_users.get(&sub.outputs[0]) else {
            continue;
        };
        let pow_id = users.iter().next().copied().unwrap();
        let pow = &model.graph.nodes[pow_id];
        if !matches!(pow.op, Op::Pow) {
            continue;
        }

        let Some(users) = value_users.get(&pow.outputs[0]) else {
            continue;
        };
        let mean2_id = users.iter().next().copied().unwrap();
        let mean2 = &model.graph.nodes[mean2_id];
        if !matches!(mean2.op, Op::ReduceMean(_)) {
            continue;
        }

        let Some(users) = value_users.get(&mean2.outputs[0]) else {
            continue;
        };
        let add_id = users.iter().next().copied().unwrap();
        let add = &model.graph.nodes[add_id];
        if !matches!(add.op, Op::Add) {
            continue;
        }

        let Some(users) = value_users.get(&add.outputs[0]) else {
            continue;
        };
        let sqrt_id = users.iter().next().copied().unwrap();
        let sqrt = &model.graph.nodes[sqrt_id];
        if !matches!(sqrt.op, Op::Sqrt) {
            continue;
        }

        let Some(users) = value_users.get(&sqrt.outputs[0]) else {
            continue;
        };
        let div_id = users.iter().next().copied().unwrap();
        let div = &model.graph.nodes[div_id];
        if !matches!(div.op, Op::Div) {
            continue;
        }

        let Some(users) = value_users.get(&div.outputs[0]) else {
            continue;
        };
        let mul_id = users.iter().next().copied().unwrap();
        let mul = &model.graph.nodes[mul_id];
        if !matches!(mul.op, Op::Mul) {
            continue;
        }

        let Some(users) = value_users.get(&mul.outputs[0]) else {
            continue;
        };
        let add2_id = users.iter().next().copied().unwrap();
        let add2 = &model.graph.nodes[add2_id];
        if !matches!(add2.op, Op::Add) {
            continue;
        }

        // LayerNormalization Detected!

        list.push((
            mean.inputs[0],
            add2.outputs[0],
            mul.inputs[1],
            add2.inputs[1],
            add.inputs[1],
        ));
        delete_list.push(mean_id);
        delete_list.push(sub_id);
        delete_list.push(pow_id);
        delete_list.push(mean2_id);
        delete_list.push(add_id);
        delete_list.push(sqrt_id);
        delete_list.push(div_id);
        delete_list.push(mul_id);
        delete_list.push(add2_id);
    }

    let count = list.len();

    for (data, end, scale, bias, epsilon) in list {
        let epsilon = model.graph.inits.get(&epsilon).unwrap().data::<f32>()[0];
        let ln_out = model.graph.values.new_val();
        let ln = Node::new(Op::LayerNormalization(LayerNormalization {
            axis: -1,
            epsilon,
            stash_type: 1,
        }))
        .with_in(data)
        .with_in(scale)
        .with_in(bias)
        .with_out(ln_out);
        let _ln_id = model.add_node(ln);

        let Some(users) = value_users.get(&end) else {
            let idx = model.graph.outputs.iter().position(|&i| i == end).unwrap();
            model.graph.outputs[idx] = ln_out;
            continue;
        };
        for user_id in users {
            let user = &mut model.graph.nodes[*user_id];
            let idx = user.inputs.iter().position(|&i| i == end).unwrap();
            user.inputs[idx] = ln_out;
        }
    }

    for node in delete_list {
        model.graph.nodes[node].deleted = true
    }

    model.remove_unnecessary_nodes();

    log::info!("fuse_layer_norm({count}): {:?}", start.elapsed());
}
