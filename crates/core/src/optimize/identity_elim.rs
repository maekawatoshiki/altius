use std::time::Instant;

use crate::{model::Model, op::Op};

pub fn eliminate_identity(model: &mut Model) {
    let start = Instant::now();

    let value_users = model.get_value_users();
    let nodes = model.topo_sort_nodes();
    for node_id in nodes {
        let node = &model.graph.nodes[node_id];
        if node.op != Op::Identity {
            continue;
        }
        assert!(node.inputs.len() == 1);
        assert!(node.outputs.len() == 1);

        let id_in = node.inputs[0];
        let id_out = node.outputs[0];
        if let Some(users) = value_users.get(&id_out) {
            for &uid in users {
                let user = &mut model.graph.nodes[uid];
                for input in &mut user.inputs {
                    if *input == id_out {
                        *input = id_in;
                    }
                }
            }
        } else {
            for output in &mut model.graph.outputs {
                if *output == id_out {
                    *output = id_in;
                }
            } 
        };

        model.graph.nodes[node_id].deleted = true;
    }

    model.remove_unnecessary_nodes();

    log::info!("eliminate_identity: {:?}", start.elapsed());
}
