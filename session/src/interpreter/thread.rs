#[cfg(target_arch = "wasm32")]
use std::marker::PhantomData;

#[cfg(not(target_arch = "wasm32"))]
use threadpool::ThreadPool;

pub struct ThreadCtx {
    #[cfg(not(target_arch = "wasm32"))]
    pub tp: ThreadPool,
}

pub struct Scope<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    tp: &'a ThreadPool,

    #[cfg(target_arch = "wasm32")]
    _phantom: PhantomData<&'a fn()>,
}

impl<'a> Scope<'a> {
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'a,
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let f = unsafe {
                std::mem::transmute::<
                    Box<dyn FnOnce() + Send + 'a>,
                    Box<dyn FnOnce() + Send + 'static>,
                >(Box::new(f))
            };
            self.tp.execute(f)
        }

        #[cfg(target_arch = "wasm32")]
        {
            f()
        }
    }
}

impl ThreadCtx {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        let tp = ThreadPool::new(1);
        tp.execute(move || core_affinity::set_for_current(core_affinity::CoreId { id: 0 }));
        tp.join();
        Self { tp }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Self {
        Self {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_with_num_threads(n: usize) -> Self {
        let tp = ThreadPool::new(n);
        for i in 0..n {
            tp.execute(move || core_affinity::set_for_current(core_affinity::CoreId { id: i }))
        }
        tp.join();
        Self { tp }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_num_threads(_n: usize) -> Self {
        Self {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn num_threads(&self) -> usize {
        self.tp.max_count()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn num_threads(&self) -> usize {
        1
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn scope<F>(&self, mut f: F)
    where
        F: FnMut(&Scope),
    {
        let scope = Scope { tp: &self.tp };
        f(&scope);
        self.tp.join();
    }

    #[cfg(target_arch = "wasm32")]
    pub fn scope<F>(&self, mut f: F)
    where
        F: FnMut(&Scope),
    {
        let scope = Scope {
            _phantom: PhantomData::default(),
        };
        f(&scope);
    }
}
