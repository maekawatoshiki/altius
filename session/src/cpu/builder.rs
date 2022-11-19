use altius_core::{model::Model, node::Node};
use rustc_hash::FxHashMap;

use crate::{create_execution_plan, infer_shapes};

use super::session::CpuSession;

pub struct CpuSessionBuilder<'a> {
    model: &'a Model,
}

impl<'a> CpuSessionBuilder<'a> {
    pub const fn new(model: &'a Model) -> Self {
        Self { model }
    }

    pub fn build(self) -> CpuSession<'a> {
        let model = self.model;

        let sorted_nodes = model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        infer_shapes(model, &sorted_nodes, &mut inferred_shapes);

        for &node_id in &sorted_nodes {
            let node = &model.nodes[node_id];

            compile(model, node);
        }

        CpuSession {
            model,
            execution_plans: create_execution_plan(model, &sorted_nodes),
        }
    }
}

fn compile(model: &Model, node: &Node) {
    todo!()
}
