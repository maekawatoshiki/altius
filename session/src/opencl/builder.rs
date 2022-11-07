use altius_core::model::Model;

use super::session::OpenclSession;

#[derive(Default)]
pub struct OpenclSessionBuilder<'a> {
    #[allow(dead_code)] // TODO: Remove later.
    model: Option<&'a Model>,
}

impl<'a> OpenclSessionBuilder<'a> {
    pub const fn new() -> Self {
        Self { model: None }
    }

    pub fn with_model(mut self, model: &'a Model) -> Self {
        self.model = Some(model);
        self
    }

    pub fn build(self) -> Option<OpenclSession<'a>> {
        let model = self.model?;
        Some(OpenclSession::new(model))
    }
}
