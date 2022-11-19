use altius_core::model::Model;

use crate::NodeExecutionPlan;

pub struct CpuSession<'a> {
    pub(super) model: &'a Model,
    pub(super) execution_plans: Vec<NodeExecutionPlan>,
}
