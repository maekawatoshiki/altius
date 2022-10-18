use threadpool::ThreadPool;

pub struct ThreadCtx {
    pub tp: ThreadPool,
}

pub struct Scope<'a> {
    tp: &'a ThreadPool,
}

impl<'a> Scope<'a> {
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'a,
    {
        let f = unsafe {
            std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Box<dyn FnOnce() + Send + 'static>>(
                Box::new(f),
            )
        };
        self.tp.execute(f)
    }
}

impl ThreadCtx {
    pub fn scope<F>(&self, mut f: F)
    where
        F: FnMut(&Scope),
    {
        let scope = Scope { tp: &self.tp };
        f(&scope);
        self.tp.join();
    }
}
