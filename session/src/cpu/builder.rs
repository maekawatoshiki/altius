use std::cell::RefCell;

use altius_core::{
    model::Model,
    node::{compute_output_shapes, Conv2d, Node, NodeId, Op},
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;

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

        let values = ThreadLocal::new();
        {
            let values = &mut *values
                .get_or(|| RefCell::new(self.model.inits.clone()))
                .borrow_mut();

            for &node_id in &sorted_nodes {
                compile(model, &inferred_shapes, values, node_id);
            }
        }

        CpuSession {
            model,
            execution_plans: create_execution_plan(model, &sorted_nodes),
            values,
        }
    }
}

fn compile(
    model: &Model,
    inferred_shapes: &FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    values: &mut FxHashMap<ValueId, Tensor>,
    node_id: NodeId,
) {
    let node = &model.nodes[node_id];
    let inputs = node
        .inputs
        .iter()
        .map(|input| values.get(input).unwrap())
        .collect::<Vec<_>>();
    // Use inferred shapes if any.
    let (op, output_shapes) = inferred_shapes.get(&node_id).cloned().unwrap_or_else(|| {
        let mut op = node.op.clone();
        let output_shapes = compute_output_shapes(&mut op, &inputs, model.opset_version);
        (op, output_shapes)
    });
    let mut outputs = output_shapes
        .into_iter()
        .map(|TypedShape { elem_ty, dims }| Tensor::empty_of_type(elem_ty, dims))
        .collect::<Vec<_>>();

    match op {
        Op::Conv2d(ref conv) => compile_conv2d(conv, &inputs, &mut outputs),
        _ => todo!(),
    }

    for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
        values.insert(val, output);
    }
}

fn compile_conv2d(conv: &Conv2d, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    todo!()
}
