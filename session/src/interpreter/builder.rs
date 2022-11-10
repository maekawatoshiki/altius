use altius_core::model::Model;

use super::InterpreterSession;

pub struct InterpreterSessionBuilder<'a> {
    model: Option<&'a Model>,
    intra_op_num_threads: usize,
    enable_profiling: bool,
}

impl<'a> Default for InterpreterSessionBuilder<'a> {
    fn default() -> Self {
        Self {
            model: None,
            intra_op_num_threads: 1,
            enable_profiling: false,
        }
    }
}

impl<'a> InterpreterSessionBuilder<'a> {
    pub const fn new() -> Self {
        Self {
            model: None,
            intra_op_num_threads: 1,
            enable_profiling: false,
        }
    }

    pub const fn with_model(mut self, model: &'a Model) -> Self {
        self.model = Some(model);
        self
    }

    pub const fn with_intra_op_num_threads(mut self, intra_op_num_threads: usize) -> Self {
        self.intra_op_num_threads = intra_op_num_threads;
        self
    }

    pub const fn with_profiling_enabled(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    pub fn build(self) -> Option<InterpreterSession<'a>> {
        let sess = InterpreterSession::new(self.model?)
            .with_profiling(self.enable_profiling)
            .with_intra_op_num_threads(self.intra_op_num_threads);
        Some(sess)
    }
}
