use std::cell::RefCell;

use altius_core::{model::Model, tensor::Tensor, value::ValueId};
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;

use crate::NodeExecutionPlan;

pub struct CpuSession<'a> {
    pub(super) model: &'a Model,
    pub(super) execution_plans: Vec<NodeExecutionPlan>,
    pub(super) values: ThreadLocal<RefCell<FxHashMap<ValueId, Tensor>>>,
}
