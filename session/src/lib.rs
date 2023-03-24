#![feature(portable_simd)]
#![allow(clippy::excessive_precision)]

pub mod interpreter;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu;
#[cfg(feature = "cpu-backend")]
pub mod cpu;

use std::borrow::Cow;

use altius_core::{analysis::shape::ShapeError, model::Model, node::NodeId, value::ValueId};
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
