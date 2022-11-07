use altius_core::model::Model;

pub struct OpenclSession<'a> {
    #[allow(dead_code)] // TODO: Remove later.
    model: &'a Model,
}

impl<'a> OpenclSession<'a> {
    pub const fn new(model: &'a Model) -> Self {
        Self { model }
    }
}
