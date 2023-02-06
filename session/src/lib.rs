#![feature(portable_simd)]
#![allow(clippy::excessive_precision)]

pub mod interpreter;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu;

use std::borrow::Cow;

use altius_core::{
    model::Model,
    node::{compute_output_shapes, NodeId, ShapeError},
    op::Op,
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
#[cfg(all(feature = "cblas", not(feature = "blis")))]
#[allow(unused)]
use blas_src;
#[cfg(all(feature = "cblas", feature = "blis"))]
#[allow(unused)]
use blis_src;
use rustc_hash::FxHashMap;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SessionError {
    /// Errors arised from shape inference.
    #[error("Shape: {0}")]
    Shape(#[from] ShapeError),

    /// General error messages (including TODOs).
    #[error("Something went wrong: {0}")]
    Message(Cow<'static, str>),
}

/// Represents a node to execute and values to be freed after the execution of the node.
#[derive(Debug)]
struct NodeExecutionPlan {
    /// The node to execute.
    node_id: NodeId,

    /// Values to be freed after the execution of the node.
    free_vals: Vec<ValueId>,
}

/// Infer `TypedShape`s of output tensors for each node.
/// It skips to infer on nodes without information for inference.
fn infer_shapes(
    model: &Model,
    sorted_nodes: &[NodeId],
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
) -> Result<(), ShapeError> {
    let mut values = model.inits.clone();

    for &val_id in &model.inputs {
        let shape = &model.values.inner()[val_id].shape;
        let Some(shape) = shape else { continue };
        let tensor = Tensor::zeros_of_type(shape.elem_ty, shape.dims.clone());
        values.insert(val_id, tensor);
    }

    for &node in sorted_nodes {
        infer_shape(model, &mut values, shapes, node)?
    }

    Ok(())
}

fn infer_shape(
    model: &Model,
    values: &mut FxHashMap<ValueId, Tensor>,
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    node_id: NodeId,
) -> Result<(), ShapeError> {
    let node = &model.nodes[node_id];
    let mut op = node.op.clone();
    let mut inputs = vec![];
    for input in &node.inputs {
        let Some(input) = values.get(input) else { return Ok(()); };
        inputs.push(input);
    }
    let output_shapes =
        compute_output_shapes(&mut op, &inputs, node.outputs.len(), model.opset_version).unwrap(); // TODO: Remove unwrap().
    let mut outputs = vec![];
    for shape in &output_shapes {
        outputs.push(Tensor::empty_of_type(shape.elem_ty, shape.dims.clone()));
    }
    for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
        values.insert(val, output);
    }
    shapes.insert(node_id, (op, output_shapes));
    Ok(())
}

fn create_execution_plan(model: &Model, sorted_nodes: &[NodeId]) -> Vec<NodeExecutionPlan> {
    let node_order: FxHashMap<NodeId, usize> = sorted_nodes
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut new_sorted_nodes = vec![];
    let mut node_to_free_vals = FxHashMap::default();
    let value_users = model.get_value_users();

    for &node_id in sorted_nodes {
        let node = &model.nodes[node_id];
        new_sorted_nodes.push(NodeExecutionPlan {
            node_id,
            free_vals: vec![],
        });

        for &output_id in &node.outputs {
            if !value_users.contains_key(&output_id) {
                continue;
            }

            let users = &value_users[&output_id];
            let last_user = users
                .iter()
                .map(|id| (node_order[id], id))
                .max_by(|x, y| x.0.cmp(&y.0))
                .unwrap()
                .1;
            node_to_free_vals
                .entry(last_user)
                .or_insert_with(Vec::new)
                .push(output_id)
        }

        if let Some(mut vals) = node_to_free_vals.remove(&node_id) {
            new_sorted_nodes
                .last_mut()
                .unwrap()
                .free_vals
                .append(&mut vals);
        }
    }

    new_sorted_nodes
}
