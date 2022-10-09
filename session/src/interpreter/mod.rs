mod conv2d;
mod gemm;

use crate::interpreter::gemm::{sgemm, sgemm2};

use super::SessionError;
use altius_core::{
    model::Model,
    node::{
        compute_output_shapes, BatchNormalization, Cast, Concat, Flatten, Gather, Gemm,
        HardSigmoid, LeakyReLU, MaxPool, Node, NodeId, Op, ReduceMean, Resize, Softmax, Squeeze,
        Transpose,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
    value::ValueId,
};
use conv2d::Conv2dCtx;
use core_affinity::CoreId;
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;
use ndarray::{s, ArrayView2, ArrayView3, ArrayView4};
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;
use threadpool::ThreadPool;

use std::{
    cell::RefCell,
    simd::{Simd, SimdFloat, SimdOrd, StdFloat},
    slice,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    pub struct SafeCudnnContext(pub CudnnContext);

    unsafe impl Send for SafeCudnnContext {}
    unsafe impl Sync for SafeCudnnContext {}
}

#[cfg(feature = "cuda")]
use cuda::*;

pub struct Interpreter<'a> {
    model: &'a Model,
    #[cfg(feature = "cuda")]
    cudnn_ctx: SafeCudnnContext,
    sorted_nodes: Vec<NodeId>,
    inferred_shapes: FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    enable_profiling: bool,
    values: ThreadLocal<RefCell<FxHashMap<ValueId, Tensor>>>,
    dummy_value: Tensor,
    tctx: ThreadCtx,
}

struct ThreadCtx {
    tp: ThreadPool,
}

#[derive(Copy, Clone, Debug)]
struct SendPtr<T: Send>(pub *const T);

#[derive(Copy, Clone, Debug)]
struct SendPtrMut<T: Send>(pub *mut T);

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Send for SendPtrMut<T> {}

impl<T: Send> SendPtr<T> {
    pub fn inner(self) -> *const T {
        self.0
    }
}

impl<T: Send> SendPtrMut<T> {
    pub fn inner(self) -> *mut T {
        self.0
    }
}

impl<'a> Interpreter<'a> {
    pub fn new(model: &'a Model) -> Self {
        let sorted_nodes = model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        infer_shapes(model, &sorted_nodes, &mut inferred_shapes);
        let workers = num_cpus::get_physical();
        let tp = ThreadPool::new(workers);
        assert_eq!(tp.queued_count(), 0);
        for p in 0..workers {
            tp.execute(move || {
                core_affinity::set_for_current(CoreId { id: p });
            });
        }
        tp.join();

        Interpreter {
            model,
            #[cfg(feature = "cuda")]
            cudnn_ctx: SafeCudnnContext(CudnnContext::new().expect("cudnn context init failed")),
            sorted_nodes,
            inferred_shapes,
            enable_profiling: false,
            values: ThreadLocal::new(),
            dummy_value: Tensor::zeros::<f32>(vec![0].into()),
            tctx: ThreadCtx { tp },
        }
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        let start = Instant::now();

        if self.model.outputs.len() > 1 {
            log::debug!("Number of outputs: {}", self.model.outputs.len());
        }

        let mut profile = FxHashMap::default();
        let values = &mut *self
            .values
            .get_or(|| RefCell::new(self.model.inits.clone()))
            .borrow_mut();

        // Set inputs.
        for (id, tensor) in inputs {
            values.insert(id, tensor);
        }

        for &node in &self.sorted_nodes {
            self.run_node(&mut profile, values, node);
        }

        if self.enable_profiling {
            log::info!(
                "Kernel execution time: {:#?}",
                profile.values().sum::<Duration>()
            );
            log::info!("Total execution time: {:#?}", start.elapsed());
            log::info!("Profile: {:#?}", profile);
        }

        Ok(self
            .model
            .outputs
            .iter()
            .map(|id| values.remove(id).unwrap())
            .collect())
    }

    fn run_node(
        &self,
        profile: &mut FxHashMap<&'static str, Duration>,
        values: &mut FxHashMap<ValueId, Tensor>,
        node_id: NodeId,
    ) {
        let node = &self.model.nodes[node_id];
        let inputs = node
            .inputs
            .iter()
            .map(|input| values.get(input).unwrap_or(&self.dummy_value))
            .collect::<Vec<_>>();
        // Use inferred shapes if any.
        let (op, output_shapes) =
            self.inferred_shapes
                .get(&node_id)
                .cloned()
                .unwrap_or_else(|| {
                    let mut op = node.op.clone();
                    let output_shapes = compute_output_shapes(&mut op, &inputs);
                    (op, output_shapes)
                });
        let mut outputs = output_shapes
            .into_iter()
            .map(|TypedShape { elem_ty, dims }| Tensor::uninit_of_type(elem_ty, dims))
            .collect::<Vec<_>>();

        let start = Instant::now();

        // Actual kernel runs here.
        match op {
            Op::Conv2d(ref conv) => conv2d::compute(&mut Conv2dCtx {
                #[cfg(feature = "cuda")]
                cudnn: &self.cudnn_ctx,
                op: conv,
                inputs: &inputs,
                outputs: &mut outputs,
            }),
            Op::Add => compute_add(&self.tctx, &inputs, &mut outputs),
            Op::Sub => compute_sub(node, &inputs, &mut outputs),
            Op::Mul => compute_mul(node, &inputs, &mut outputs),
            Op::Div => compute_div(&self.tctx, &inputs, &mut outputs),
            Op::Pow => compute_pow(node, &inputs, &mut outputs),
            Op::Sqrt => compute_sqrt(node, &inputs, &mut outputs),
            Op::MaxPool(ref maxpool) => compute_max_pool(maxpool, &inputs, &mut outputs),
            Op::GlobalAveragePool => compute_gavg_pool(node, &inputs, &mut outputs),
            Op::Reshape => compute_reshape(node, &inputs, &mut outputs),
            Op::Flatten(ref flatten) => compute_flatten(flatten, &inputs, &mut outputs),
            Op::MatMul => compute_mat_mul(node, &inputs, &mut outputs),
            Op::Gemm(ref gemm) => compute_gemm(gemm, &inputs, &mut outputs),
            Op::ReLU => compute_relu(node, &inputs, &mut outputs),
            Op::HardSigmoid(ref hs) => compute_hard_sigmoid(hs, &inputs, &mut outputs),
            Op::LeakyReLU(ref leaky) => compute_leaky_relu(leaky, &inputs, &mut outputs),
            Op::Gelu => compute_gelu(&self.tctx, &inputs, &mut outputs),
            Op::Sigmoid => compute_sigmoid(&inputs, &mut outputs),
            Op::Erf => compute_erf(&inputs, &mut outputs),
            Op::Clip => todo!("clip"),
            Op::Softmax(ref softmax) => compute_softmax(&self.tctx, softmax, &inputs, &mut outputs),
            Op::Resize(ref resize) => compute_resize(resize, &inputs, &mut outputs),
            Op::Concat(ref concat) => compute_concat(concat, &inputs, &mut outputs),
            Op::Transpose(ref trans) => compute_transpose(trans, &inputs, &mut outputs),
            Op::Squeeze(ref squeeze) => compute_squeeze(squeeze, &inputs, &mut outputs),
            Op::Unsqueeze(_) => todo!("unsqueeze"),
            Op::ReduceMin(_) => todo!("reduce min"),
            Op::ReduceMean(ref rmean) => compute_reduce_mean(rmean, &inputs, &mut outputs),
            Op::Round => todo!("round"),
            Op::Exp => todo!("exp"),
            Op::Loop => compute_loop(node, &inputs, &mut outputs),
            Op::Tile => compute_tile(node, &inputs, &mut outputs),
            Op::Cast(ref cast) => compute_cast(cast, &inputs, &mut outputs),
            Op::BatchNormalization(ref batchnorm) => {
                compute_batch_normalization(batchnorm, &inputs, &mut outputs)
            }
            Op::Slice => compute_slice(node, &inputs, &mut outputs),
            Op::Gather(ref gather) => compute_gather(gather, &inputs, &mut outputs),
            Op::Shape(_) => todo!("shape"),
            Op::NonMaxSuppression => todo!("nms"),
            Op::Constant(_) => todo!("constant"),
        }

        if self.enable_profiling {
            let elapsed = start.elapsed();
            *profile.entry(op.name()).or_insert(Duration::ZERO) += elapsed;
        }

        for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
            values.insert(val, output);
        }
    }
}

fn compute_gavg_pool(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::GLOBALAVERAGEPOOL_IN];
    let output = &mut outputs[Node::GLOBALAVERAGEPOOL_OUT];

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    let n = input.dims()[0];
    let c = input.dims()[1];
    let h = input.dims()[2];
    let w = input.dims()[3];
    let area = (h * w) as f32;
    let input_strides = input.strides();
    let input = input.data::<f32>();
    let out_stride_n = output.strides()[0];
    let output = output.data_mut::<f32>();

    for n in 0..n {
        let idx_n = input_strides[0] * n;
        for c in 0..c {
            let idx_c = idx_n + input_strides[1] * c;
            let mut sum = 0f32;
            for h in 0..h {
                let idx_h = idx_c + input_strides[2] * h;
                for w in 0..w {
                    let idx_w = idx_h + input_strides[3] * w;
                    sum += input[idx_w];
                }
            }
            output[n * out_stride_n + c] = sum / area;
        }
    }
}

fn compute_max_pool(maxpool: &MaxPool, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::MAXPOOL_IN];
    let output = &mut outputs[Node::MAXPOOL_OUT];

    let kernel = &maxpool.kernel_shape;
    let stride = &maxpool.strides;

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    let batches = output.dims()[0];
    let channels = output.dims()[1];
    let outer = batches * channels;
    let output_h = output.dims()[2];
    let output_w = output.dims()[3];
    let input_h = input.dims()[2];
    let input_w = input.dims()[3];
    let kernel_h = kernel[0];
    let kernel_w = kernel[1];
    let stride_h = stride[0];
    let stride_w = stride[1];
    let input_hw = input_h * input_w;
    let output_hw = output_h * output_w;

    let mut input = input.data::<f32>();
    let mut output = output.data_mut::<f32>();

    for _ in 0..outer {
        let mut y = 0isize; // TODO: pad
        for ay in 0..output_h {
            let mut x = 0isize; // TODO: pad
            let output = &mut output[ay * output_w..];
            for ax in 0..output_w {
                let mut max = f32::MIN;
                for fy in 0..kernel_h {
                    let oy = y + fy as isize;
                    let input = &input[oy as usize * input_w..];
                    for fx in 0..kernel_w {
                        let ox = x + fx as isize;
                        if ox < 0 || oy < 0 || ox >= input_w as isize || oy >= input_h as isize {
                            continue;
                        }
                        max = input[ox as usize].max(max);
                    }
                }
                output[ax] = if max == f32::MIN { 0.0 } else { max };
                x += stride_w as isize
            }
            y += stride_h as isize
        }
        input = &input[input_hw..];
        output = &mut output[output_hw..];
    }
}

fn compute_add(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::ADD_IN_A];
    let input_b = inputs[Node::ADD_IN_B];
    let output = &mut outputs[Node::ADD_OUT];

    let adims = input_a.dims();
    let bdims = input_b.dims();

    if adims == bdims {
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();
        const SIMD_LEN: usize = 4;
        let chunk = 100000;

        input_a
            .chunks(chunk)
            .zip(input_b.chunks(chunk))
            .zip(output.chunks_mut(chunk))
            .for_each(|((a, b), o)| {
                let mut len = a.len();
                let a = SendPtr(a.as_ptr());
                let b = SendPtr(b.as_ptr());
                let o = SendPtrMut(o.as_mut_ptr());

                tctx.tp.execute(move || {
                    let mut input_a = unsafe { slice::from_raw_parts(a.inner(), len) };
                    let mut input_b = unsafe { slice::from_raw_parts(b.inner(), len) };
                    let mut output = unsafe { slice::from_raw_parts_mut(o.inner(), len) };

                    while len >= SIMD_LEN {
                        let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
                        let b = Simd::<f32, SIMD_LEN>::from_slice(&input_b[0..SIMD_LEN]);
                        output[0..SIMD_LEN].copy_from_slice((a + b).as_ref());
                        input_a = &input_a[SIMD_LEN..];
                        input_b = &input_b[SIMD_LEN..];
                        output = &mut output[SIMD_LEN..];
                        len -= SIMD_LEN
                    }

                    for ((a, b), o) in input_a.iter().zip(input_b.iter()).zip(output.iter_mut()) {
                        *o = a + b;
                    }
                });
            });

        tctx.tp.join();

        return;
    }

    if adims.len() == 4 && bdims.len() == 3 {
        assert!(adims[1] == bdims[0]);
        assert!(bdims[1] == 1);
        assert!(bdims[2] == 1);

        for n in 0..adims[0] {
            for z in 0..adims[1] {
                for x in 0..adims[2] {
                    for y in 0..adims[3] {
                        *output.at_4d_mut(n, z, x, y) =
                            input_a.at_4d(n, z, x, y) + input_b.at_3d(z, 0, 0);
                    }
                }
            }
        }

        return;
    }

    if adims.len() == bdims.len() && bdims[bdims.len() - 1] == 1 {
        let dims = adims;
        let max = dims.total_elems();
        let n = dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        for i in 0..max {
            output[i] = input_a[i] - input_b[i / n];
        }

        return;
    }

    if bdims.is_scalar() {
        let b = input_b.data::<f32>()[0];
        let output = output.data_mut::<f32>();

        for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
            *o = a + b;
        }

        return;
    }

    let (_adims, bdims, input_a, input_b) =
        if bdims.len() == 1 && adims[adims.len() - 1] == bdims[0] {
            (adims, bdims, input_a, input_b)
        } else if adims.len() == 1 && bdims[bdims.len() - 1] == adims[0] {
            (bdims, adims, input_b, input_a)
        } else {
            todo!("A shape: {:?}, B shape: {:?}", adims, bdims)
        };

    let input_a = input_a.data::<f32>();
    let output = output.data_mut::<f32>();
    let blen = bdims[0];
    const SIMD_LEN: usize = 4;

    let input_b_ptr = SendPtr(input_b.data::<f32>().as_ptr());
    let batch = (100000 / blen).max(1);

    input_a
        .chunks(blen * batch)
        .zip(output.chunks_mut(blen * batch))
        .for_each(|(input_a, output)| {
            let out_len = output.len();
            let input_a_ptr = SendPtr(input_a.as_ptr());
            let output_ptr = SendPtrMut(output.as_mut_ptr());

            tctx.tp.execute(move || {
                let input_b = unsafe { slice::from_raw_parts(input_b_ptr.inner(), blen) };

                for i in 0..batch {
                    if i * blen >= out_len {
                        break;
                    }

                    let mut len = blen;
                    let mut input_b = input_b;
                    let mut output = unsafe {
                        slice::from_raw_parts_mut(output_ptr.inner().add(i * blen), blen)
                    };
                    let mut input_a =
                        unsafe { slice::from_raw_parts(input_a_ptr.inner().add(i * blen), blen) };

                    while len >= SIMD_LEN {
                        let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
                        let b = Simd::<f32, SIMD_LEN>::from_slice(&input_b[0..SIMD_LEN]);
                        output[0..SIMD_LEN].copy_from_slice((a + b).as_ref());
                        input_a = &input_a[SIMD_LEN..];
                        input_b = &input_b[SIMD_LEN..];
                        output = &mut output[SIMD_LEN..];
                        len -= SIMD_LEN
                    }

                    for ((a, b), o) in input_a.iter().zip(input_b.iter()).zip(output.iter_mut()) {
                        *o = a + b;
                    }
                }
            })
        });

    tctx.tp.join();
}

fn compute_sub(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::SUB_IN_A];
    let input_b = inputs[Node::SUB_IN_B];
    let output = &mut outputs[Node::SUB_OUT];

    if input_a.dims() == input_b.dims() {
        let mut input_a = input_a.data::<f32>();
        let mut input_b = input_b.data::<f32>();
        let mut output = output.data_mut::<f32>();
        let mut len = output.len();
        const SIMD_LEN: usize = 4;

        while len >= SIMD_LEN {
            let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
            let b = Simd::<f32, SIMD_LEN>::from_slice(&input_b[0..SIMD_LEN]);
            let c = a - b;
            output[0..SIMD_LEN].copy_from_slice(c.as_array());
            input_a = &input_a[SIMD_LEN..];
            input_b = &input_b[SIMD_LEN..];
            output = &mut output[SIMD_LEN..];
            len -= SIMD_LEN
        }

        for (i, (a, b)) in input_a.iter().zip(input_b.iter()).enumerate() {
            output[i] = a - b;
        }

        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_a.dims().len() == 3
        && input_b.dims().len() == 3
        && input_b.dims()[input_b.dims().len() - 1] == 1
    {
        let dims = input_a.dims();
        let max = dims.total_elems();
        let n = *dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        input_a
            .chunks(n)
            .zip(input_b.iter())
            .zip(output.chunks_mut(n))
            .for_each(|((a, b), o)| {
                for (a, o) in a.iter().zip(o.iter_mut()) {
                    *o = a - b;
                }
            });

        return;
    }

    todo!(
        "A shape: {:?}, B shape: {:?}",
        input_a.dims(),
        input_b.dims()
    )
}

fn compute_mul(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::MUL_IN_A];
    let input_b = inputs[Node::MUL_IN_B];
    let output = &mut outputs[Node::MUL_OUT];

    let adims = input_a.dims();
    let bdims = input_b.dims();

    if adims == bdims {
        let output = output.data_mut::<f32>();
        for (i, (a, b)) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .enumerate()
        {
            output[i] = a * b;
        }

        return;
    }

    // TODO: We need multidirectional broadcast!

    let in_a = adims;
    let in_b = bdims;
    if in_a.len() == 4
        && in_b.len() == 4
        && in_a[0] == in_b[0]
        && in_a[1] == in_b[1]
        && in_a[2] == 1
        && in_a[3] == 1
    {
        let input_a_strides = input_a.strides();
        let output_strides = output.strides().to_vec();
        let n = in_a[0];
        let z = in_a[1];
        let b_x = in_b[2];
        let b_y = in_b[3];
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        for n in 0..n {
            let o_s_n = n * output_strides[0];
            let ia_s_n = n * input_a_strides[0];
            for z in 0..z {
                let o_s_z = o_s_n + z * output_strides[1];
                let ia_s_z = ia_s_n + z * input_a_strides[1];
                let d = input_a[ia_s_z];
                for x in 0..b_x {
                    let o_s_x = o_s_z + x * output_strides[2];
                    for y in 0..b_y {
                        let o_s_y = o_s_x + y;
                        output[o_s_y] = d * input_b[o_s_y];
                    }
                }
            }
        }

        return;
    }

    if bdims.is_scalar() {
        let b = input_b.data::<f32>()[0];
        let output = output.data_mut::<f32>();

        for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
            *o = a * b;
        }

        return;
    }

    let (adims, bdims, input_a, input_b) = if bdims.len() == 1 && adims[adims.len() - 1] == bdims[0]
    {
        (adims, bdims, input_a, input_b)
    } else if adims.len() == 1 && bdims[bdims.len() - 1] == adims[0] {
        (bdims, adims, input_b, input_a)
    } else {
        todo!("A shape: {:?}, B shape: {:?}", adims, bdims)
    };

    let total = adims.total_elems();
    let part = bdims[0];
    let b = input_b.data::<f32>().as_ptr();
    let mut a = input_a.data::<f32>().as_ptr();
    let mut output = output.data_mut::<f32>().as_mut_ptr();

    for _ in 0..total / part {
        let mut b = b;
        for _ in 0..part {
            unsafe { *output = *a * *b };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            output = unsafe { output.add(1) };
        }
    }
}

fn compute_div(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::MUL_IN_A];
    let input_b = inputs[Node::MUL_IN_B];
    let output = &mut outputs[Node::MUL_OUT];

    if input_a.dims() == input_b.dims() {
        for (i, (a, b)) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .enumerate()
        {
            output.data_mut::<f32>()[i] = a / b;
        }
        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_a.dims().len() == 3
        && input_b.dims().len() == 3
        && input_b.dims()[input_b.dims().len() - 1] == 1
    {
        {
            let dims = input_a.dims();
            let n = dims.0.last().unwrap();
            let input_a = input_a.data::<f32>();
            let input_b = input_b.data::<f32>();
            let output = output.data_mut::<f32>();

            input_a
                .chunks(*n)
                .zip(input_b.iter())
                .zip(output.chunks_mut(*n))
                .for_each(|((a, &b), o)| {
                    let len = a.len();
                    let a = SendPtr(a.as_ptr());
                    let o = SendPtrMut(o.as_mut_ptr());
                    tctx.tp.execute(move || {
                        let a = unsafe { slice::from_raw_parts(a.inner(), len) };
                        let o = unsafe { slice::from_raw_parts_mut(o.inner(), len) };
                        for (&a, o) in a.iter().zip(o.iter_mut()) {
                            *o = a / b;
                        }
                    });
                });

            tctx.tp.join();
        }

        return;
    }

    if input_b.dims().is_scalar() {
        let b = input_b.data::<f32>()[0];
        let output = output.data_mut::<f32>();

        for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
            *o = a / b;
        }

        return;
    }

    todo!(
        "A shape: {:?}, B shape: {:?}",
        input_a.dims(),
        input_b.dims()
    )
}

fn compute_pow(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[0];
    let input_b = inputs[1];
    let output = &mut outputs[0].data_mut();

    if input_a.dims() == input_b.dims() {
        for (i, (a, b)) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .enumerate()
        {
            output[i] = a.powf(*b);
        }
        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_b.dims().is_scalar() {
        let b = input_b.data::<f32>()[0];

        if b == 2. {
            for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
                *o = a * a;
            }
        } else {
            for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
                *o = a.powf(b);
            }
        }

        return;
    }

    todo!(
        "A shape: {:?}, B shape: {:?}",
        input_a.dims(),
        input_b.dims()
    )
}

fn compute_sqrt(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0];
    let output = &mut outputs[0];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = i.sqrt();
    }
}

fn compute_mat_mul(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::MATMUL_IN_A];
    let input_b = inputs[Node::MATMUL_IN_B];
    let output = &mut outputs[Node::MATMUL_OUT];

    let adim = input_a.dims();
    let bdim = input_b.dims();

    assert!(
        (adim.len() == 2 && bdim.len() == 2 && adim[1] == bdim[0])
            || (adim.len() == 3 && bdim.len() == 2 && adim[2] == bdim[0])
            || (adim.len() == 3 && bdim.len() == 3 && adim[0] == bdim[0] && adim[2] == bdim[1]),
        "A shape: {adim:?}, B shape: {bdim:?}"
    );

    if adim.len() == 3 && bdim.len() == 2 {
        let [batch, m, _k] = input_a.fixed_dims::<3>();
        let [k, n] = input_b.fixed_dims::<2>();
        let a = input_a.data::<f32>();
        let b = input_b.data::<f32>();
        let c = output.data_mut::<f32>();

        sgemm(batch * m, k, n, 1., a, k, b, n, 0., c, n);
    } else if adim.len() == 3 && bdim.len() == 3 {
        let [_batch, m, _k] = input_a.fixed_dims::<3>();
        let [_batch, k, n] = input_b.fixed_dims::<3>();
        // TODO: Are there a better way?
        output
            .data_mut::<f32>()
            .chunks_mut(m * n)
            .zip(input_a.data::<f32>().chunks(m * k))
            .zip(input_b.data::<f32>().chunks(k * n))
            .for_each(|((c, a), b)| {
                sgemm(m, k, n, 1., a, k, b, n, 0., c, n);
            });
    } else {
        let [m, _k] = input_a.fixed_dims::<2>();
        let [k, n] = input_b.fixed_dims::<2>();

        let a = input_a.data();
        let b = input_b.data();
        let c = output.data_mut();
        sgemm(m, k, n, 1., a, k, b, n, 0., c, n);
    }
}

fn compute_gemm(gemm: &Gemm, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::GEMM_IN_A];
    let input_b = inputs[Node::GEMM_IN_B];
    let input_c = inputs[Node::GEMM_IN_C];
    let output = &mut outputs[Node::GEMM_OUT];

    assert!(input_a.dims().len() == 2);
    assert!(input_b.dims().len() == 2);
    assert!(input_c.dims().len() == 1);

    #[cfg(feature = "cblas")]
    {
        let m = input_a.dims()[gemm.trans_a as usize];
        let k = input_a.dims()[1 - gemm.trans_a as usize];
        let n = input_b.dims()[1 - gemm.trans_b as usize];

        let a = input_a.data();
        let b = input_b.data();
        let c = output.data_mut::<f32>();

        c.chunks_mut(input_c.dims()[0])
            .for_each(|o| o.copy_from_slice(input_c.data::<f32>()));

        sgemm2(
            gemm.trans_a,
            gemm.trans_b,
            m,
            k,
            n,
            gemm.alpha,
            a,
            if gemm.trans_a { m } else { k },
            b,
            if gemm.trans_b { k } else { n },
            gemm.beta,
            c,
            n,
        );
    }

    #[cfg(not(feature = "cblas"))]
    {
        use ndarray::Array2;

        let a = Array2::from_shape_vec(input_a.fixed_dims::<2>(), input_a.data::<f32>().to_vec())
            .unwrap();
        let b = Array2::from_shape_vec(input_b.fixed_dims::<2>(), input_b.data::<f32>().to_vec())
            .unwrap();
        let a = if gemm.trans_a { a.t() } else { a.view() };
        let b = if gemm.trans_b { b.t() } else { b.view() };

        let c = Array2::from_shape_vec([1, input_c.dims()[0]], input_c.data::<f32>().to_vec())
            .unwrap()
            .broadcast(output.fixed_dims::<2>())
            .unwrap()
            .into_owned();
        let mut c = c.as_standard_layout();

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let a = a.as_standard_layout();
        let a = a.as_slice().unwrap();
        let b = b.as_standard_layout();
        let b = b.as_slice().unwrap();

        sgemm(
            m,
            k,
            n,
            gemm.alpha,
            a,
            k,
            b,
            n,
            gemm.beta,
            c.as_slice_mut().unwrap(),
            n,
        );

        output.set_raw_vec(c.into_owned().into_raw_vec())
    }
}

fn compute_relu(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::RELU_IN];
    let output = &mut outputs[Node::RELU_OUT];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = i.max(0.0);
    }
}

fn compute_hard_sigmoid(hs: &HardSigmoid, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::HARDSIGMOID_IN];
    let output = &mut outputs[Node::HARDSIGMOID_OUT];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = (hs.alpha * i + hs.beta).min(1.0).max(0.0);
    }
}

fn compute_leaky_relu(leaky: &LeakyReLU, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[Node::HARDSIGMOID_IN].data();
    let output: &mut [f32] = outputs[Node::HARDSIGMOID_OUT].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = if *i < 0.0 { leaky.alpha * i } else { *i };
    }
}

fn compute_gelu(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    const B: f32 = 0.7978845608028654f32; // sqrt(2.0 / PI)
    const C: f32 = 0.035677408136300125f32; // 0.044715 * sqrt(2.0 / PI)

    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();
    let n = tctx.tp.max_count();

    input
        .chunks(input.len() / n)
        .zip(output.chunks_mut(input.len() / n))
        .for_each(|(input, output)| {
            let len = input.len();
            let input_ptr = SendPtr(input.as_ptr());
            let output_ptr = SendPtrMut(output.as_mut_ptr());
            tctx.tp.execute(move || {
                let input = unsafe { slice::from_raw_parts(input_ptr.inner(), len) };
                let output = unsafe { slice::from_raw_parts_mut(output_ptr.inner(), len) };
                for (&i, o) in input.iter().zip(output.iter_mut()) {
                    *o = i * (C * i * i + B);
                }
                tanh(output);
                for (&i, o) in input.iter().zip(output.iter_mut()) {
                    *o = (*o + 1.) * (i * 0.5);
                }
            })
        });

    tctx.tp.join();
}

fn tanh(mut data: &mut [f32]) {
    let lower_range = -9f32;
    let upper_range = 9f32;
    let alpha_13 = -2.76076847742355e-16f32;
    let alpha_11 = 2.00018790482477e-13f32;
    let alpha_9 = -8.60467152213735e-11f32;
    let alpha_7 = 5.12229709037114e-08f32;
    let alpha_5 = 1.48572235717979e-05f32;
    let alpha_3 = 6.37261928875436e-04f32;
    let alpha_1 = 4.89352455891786e-03f32;
    let beta_6 = 1.19825839466702e-06f32;
    let beta_4 = 1.18534705686654e-04f32;
    let beta_2 = 2.26843463243900e-03f32;
    let beta_0 = 4.89352518554385e-03f32;

    const SIMD_LEN: usize = 4;

    let lower_ranges = Simd::<f32, SIMD_LEN>::from_array([lower_range; SIMD_LEN]);
    let upper_ranges = Simd::<f32, SIMD_LEN>::from_array([upper_range; SIMD_LEN]);
    let alpha_13s = Simd::<f32, SIMD_LEN>::from_array([alpha_13; SIMD_LEN]);
    let alpha_11s = Simd::<f32, SIMD_LEN>::from_array([alpha_11; SIMD_LEN]);
    let alpha_9s = Simd::<f32, SIMD_LEN>::from_array([alpha_9; SIMD_LEN]);
    let alpha_7s = Simd::<f32, SIMD_LEN>::from_array([alpha_7; SIMD_LEN]);
    let alpha_5s = Simd::<f32, SIMD_LEN>::from_array([alpha_5; SIMD_LEN]);
    let alpha_3s = Simd::<f32, SIMD_LEN>::from_array([alpha_3; SIMD_LEN]);
    let alpha_1s = Simd::<f32, SIMD_LEN>::from_array([alpha_1; SIMD_LEN]);
    let beta_6s = Simd::<f32, SIMD_LEN>::from_array([beta_6; SIMD_LEN]);
    let beta_4s = Simd::<f32, SIMD_LEN>::from_array([beta_4; SIMD_LEN]);
    let beta_2s = Simd::<f32, SIMD_LEN>::from_array([beta_2; SIMD_LEN]);
    let beta_0s = Simd::<f32, SIMD_LEN>::from_array([beta_0; SIMD_LEN]);

    let mut len = data.len();

    while len >= SIMD_LEN {
        let vals = Simd::<f32, SIMD_LEN>::from_slice(&data[0..SIMD_LEN]);
        let vals = lower_ranges.simd_max(vals);
        let vals = upper_ranges.simd_min(vals);

        let vals_squared = vals * vals;

        let p = vals_squared.mul_add(alpha_13s, alpha_11s);
        let p = p.mul_add(vals_squared, alpha_9s);
        let p = p.mul_add(vals_squared, alpha_7s);
        let p = p.mul_add(vals_squared, alpha_5s);
        let p = p.mul_add(vals_squared, alpha_3s);
        let p = p.mul_add(vals_squared, alpha_1s);
        let p = p * vals;

        let q = vals_squared.mul_add(beta_6s, beta_4s);
        let q = q.mul_add(vals_squared, beta_2s);
        let q = q.mul_add(vals_squared, beta_0s);

        data[0..SIMD_LEN].copy_from_slice((p / q).as_ref());

        len -= SIMD_LEN;
        data = &mut data[SIMD_LEN..];
    }

    for x in data {
        let val = *x;
        let val = if val < lower_range { lower_range } else { val };
        let val = if val > upper_range { upper_range } else { val };

        let val_squared = val * val;

        let p = val_squared.mul_add(alpha_13, alpha_11);
        let p = p.mul_add(val_squared, alpha_9);
        let p = p.mul_add(val_squared, alpha_7);
        let p = p.mul_add(val_squared, alpha_5);
        let p = p.mul_add(val_squared, alpha_3);
        let p = p.mul_add(val_squared, alpha_1);
        let p = p * val;

        let q = val_squared.mul_add(beta_6, beta_4);
        let q = q.mul_add(val_squared, beta_2);
        let q = q.mul_add(val_squared, beta_0);

        *x = p / q
    }
}

fn exp(mut output: &mut [f32], mut input: &[f32]) {
    let lower_range = -103.9720840454f32;
    let upper_range = 88.7762626647950f32;
    // let lower_range_sum_exp = -88.3762626647949f32;
    // let upper_range_sum_exp = 88.3762626647949f32;
    let rounding_bias = 12582912.0f32;
    let log2reciprocal = 1.44269504088896341f32;
    let log2high = -6.93145752e-1f32;
    let log2low = -1.42860677e-6f32;
    let poly_0 = 0.0013780593872f32;
    let poly_1 = 0.0083731245250f32;
    let poly_2 = 0.0416695363820f32;
    let poly_3 = 0.1666647195816f32;
    let poly_4 = 0.4999998509884f32;
    let poly_56 = 1.0000000000000f32;
    let minimum_exponent = -1056964608i32;
    let maximum_exponent = 0x3F800000i32;

    const SIMD_LEN: usize = 4;

    let lower_ranges = Simd::<f32, SIMD_LEN>::from_array([lower_range; SIMD_LEN]);
    let upper_ranges = Simd::<f32, SIMD_LEN>::from_array([upper_range; SIMD_LEN]);
    // let lower_range_sum_exp = Simd::<f32, SIMD_LEN>::from_array([lower_range_sum_exp; SIMD_LEN]);
    // let upper_range_sum_exp = Simd::<f32, SIMD_LEN>::from_array([upper_range_sum_exp; SIMD_LEN]);
    let rounding_biases = Simd::<f32, SIMD_LEN>::from_array([rounding_bias; SIMD_LEN]);
    let log2reciprocals = Simd::<f32, SIMD_LEN>::from_array([log2reciprocal; SIMD_LEN]);
    let log2highs = Simd::<f32, SIMD_LEN>::from_array([log2high; SIMD_LEN]);
    let log2lows = Simd::<f32, SIMD_LEN>::from_array([log2low; SIMD_LEN]);
    let poly_0s = Simd::<f32, SIMD_LEN>::from_array([poly_0; SIMD_LEN]);
    let poly_1s = Simd::<f32, SIMD_LEN>::from_array([poly_1; SIMD_LEN]);
    let poly_2s = Simd::<f32, SIMD_LEN>::from_array([poly_2; SIMD_LEN]);
    let poly_3s = Simd::<f32, SIMD_LEN>::from_array([poly_3; SIMD_LEN]);
    let poly_4s = Simd::<f32, SIMD_LEN>::from_array([poly_4; SIMD_LEN]);
    let poly_56s = Simd::<f32, SIMD_LEN>::from_array([poly_56; SIMD_LEN]);
    let shl23 = Simd::<i32, SIMD_LEN>::from_array([23; SIMD_LEN]);
    let maximum_exponents = Simd::<i32, SIMD_LEN>::from_array([maximum_exponent; SIMD_LEN]);
    let minimum_exponents = Simd::<i32, SIMD_LEN>::from_array([minimum_exponent; SIMD_LEN]);

    let mut len = input.len();

    while len >= SIMD_LEN {
        let vals = Simd::<f32, SIMD_LEN>::from_slice(&input[0..SIMD_LEN]);
        let vals = lower_ranges.simd_max(vals);
        let vals = upper_ranges.simd_min(vals);

        let biased = vals.mul_add(log2reciprocals, rounding_biases);
        let m = biased - rounding_biases;

        let vals = m.mul_add(log2highs, vals);
        let vals = m.mul_add(log2lows, vals);

        let overflow: Simd<i32, SIMD_LEN> =
            unsafe { std::mem::transmute::<_, Simd<i32, SIMD_LEN>>(biased) } << shl23;
        let normal = overflow.simd_min(maximum_exponents);
        let normal = normal.simd_max(minimum_exponents);
        let overflow = overflow - normal;
        let overflow = overflow + maximum_exponents;
        let normal = normal + maximum_exponents;

        let p = poly_0s;
        let p = p.mul_add(vals, poly_1s);
        let p = p.mul_add(vals, poly_2s);
        let p = p.mul_add(vals, poly_3s);
        let p = p.mul_add(vals, poly_4s);
        let p = p.mul_add(vals, poly_56s);

        let vals = vals * unsafe { std::mem::transmute::<_, Simd<f32, SIMD_LEN>>(overflow) };
        let p = p.mul_add(vals, unsafe {
            std::mem::transmute::<_, Simd<f32, SIMD_LEN>>(overflow)
        });
        let p = p * unsafe { std::mem::transmute::<_, Simd<f32, SIMD_LEN>>(normal) };

        output[0..SIMD_LEN].copy_from_slice(p.as_ref());

        len -= SIMD_LEN;
        input = &input[SIMD_LEN..];
        output = &mut output[SIMD_LEN..];
    }

    for (i, o) in input.iter().zip(output.iter_mut()) {
        let val = *i;
        let val = lower_range.max(val);
        let val = upper_range.min(val);

        let biased = val * log2reciprocal + rounding_bias;
        let m = biased - rounding_bias;

        let val = m * log2high + val;
        let val = m * log2low + val;

        // overflow
        let overflow = unsafe { std::mem::transmute::<_, i32>(biased) } << 23i32;
        let normal = overflow.min(maximum_exponent);
        let normal = normal.max(minimum_exponent);
        let overflow = overflow - normal;
        let overflow = overflow + maximum_exponent;
        let normal = normal + maximum_exponent;

        let p = poly_0;
        let p = p * val + poly_1;
        let p = p * val + poly_2;
        let p = p * val + poly_3;
        let p = p * val + poly_4;
        let p = p * val + poly_56;

        let val = val * unsafe { std::mem::transmute::<_, f32>(overflow) };
        let p = p * val + unsafe { std::mem::transmute::<_, f32>(overflow) };
        let p = p * unsafe { std::mem::transmute::<_, f32>(normal) };

        *o = p;
    }
}

fn compute_sigmoid(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[Node::SIGMOID_IN].data();
    let output: &mut [f32] = outputs[Node::SIGMOID_OUT].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = 1. / (1. + (-i).exp())
    }
}

fn compute_erf(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();

    for (&i, o) in input.iter().zip(output.iter_mut()) {
        *o = fastapprox::faster::erf(i);
    }
}

fn compute_softmax(
    tctx: &ThreadCtx,
    softmax: &Softmax,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) {
    let input = inputs[0];
    let output = &mut outputs[0];

    assert_eq!(input.dims().len(), 3);
    assert_eq!(softmax.axis, -1);

    let axis_len = input.dims()[2];
    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();

    let n = tctx.tp.max_count();
    let chunk = if input.len() < n {
        input.len()
    } else {
        input.len() / n
    };

    input
        .chunks(chunk)
        .zip(output.chunks_mut(chunk))
        .for_each(|(input, output)| {
            let len = input.len();
            let input_ptr = SendPtr(input.as_ptr());
            let output_ptr = SendPtrMut(output.as_mut_ptr());
            tctx.tp.execute(move || {
                let input = unsafe { slice::from_raw_parts(input_ptr.inner(), len) };
                let output = unsafe { slice::from_raw_parts_mut(output_ptr.inner(), len) };
                exp(output, input)
            })
        });

    tctx.tp.join();

    let batch = (output.len() / 100000).max(1); // 100000 is magic number :(
                                                // I think processing more than 100000 elements for
                                                // each core is just right.

    output.chunks_mut(axis_len * batch).for_each(|output| {
        let output_ptr = SendPtrMut(output.as_mut_ptr());
        tctx.tp.execute(move || {
            for i in 0..batch {
                let output = unsafe {
                    slice::from_raw_parts_mut(output_ptr.inner().add(i * axis_len), axis_len)
                };
                let sum: f32 = output.iter().sum();
                let rsum = 1. / sum;
                for o in output.iter_mut() {
                    *o *= rsum;
                }
            }
        })
    });

    tctx.tp.join();
}

fn compute_resize(_resize: &Resize, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    log::info!("Resize: Current implementation uses bilinear interpolation!");

    assert_eq!(inputs.len(), 4);

    let input = inputs[Node::RESIZE_IN_X];
    // let sizes = inputs[Node::RESIZE_IN_SIZES];
    let output = &mut outputs[Node::RESIZE_OUT];

    let batch_size = input.dims()[0];
    let input_c = input.dims()[1];
    let input_h = input.dims()[2];
    let input_w = input.dims()[3];
    let output_h = output.dims()[2];
    let output_w = output.dims()[3];
    let input_hw = input_h * input_w;
    let outer = batch_size * input_c;

    let scale = output_h as f32 / input_h as f32;

    let mut input = input.data::<f32>();
    let mut output = output.data_mut::<f32>();

    for _ in 0..outer {
        for h in 0..output_h {
            let ihf = (h as f32 / scale - 0.5).max(0.);
            let ih = ihf as usize;
            let ih0 = ih.min(input_h - 1);
            let ih1 = (ih + 1).min(input_h - 1);
            let ih0w = input_w * ih0;
            let ih1w = input_w * ih1;
            for w in 0..output_w {
                let iwf = (w as f32 / scale - 0.5).max(0.);
                let iw = iwf as usize;
                let iw0 = iw.min(input_w - 1);
                let iw1 = (iw + 1).min(input_w - 1);

                let v00 = input[ih0w + iw0];
                let v01 = input[ih0w + iw1];
                let v10 = input[ih1w + iw0];
                let v11 = input[ih1w + iw1];

                let hd = v00 + (v10 - v00) * (ihf - ih as f32);
                let hw = v01 + (v11 - v01) * (ihf - ih as f32);
                let r = hd + (hw - hd) * (iwf - iw as f32);

                output[0] = r;
                output = &mut output[1..];
            }
        }
        input = &input[input_hw..];
    }
}

fn compute_concat(concat: &Concat, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let output = &mut outputs[Node::CONCAT_OUT];

    assert!(matches!(output.dims().len(), 3 | 4));
    assert_eq!(concat.axis, 1);

    let mut output = output.data_mut::<f32>();
    for input in inputs {
        let input = input.data::<f32>();
        output[0..input.len()].copy_from_slice(input);
        output = &mut output[input.len()..];
    }
}

fn compute_transpose(transpose: &Transpose, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::TRANSPOSE_IN];
    let output = &mut outputs[Node::TRANSPOSE_OUT];
    assert!(input.elem_ty().is_f32());

    // TODO: Refactor.
    match input.dims().len() {
        2 => {
            let in_view =
                ArrayView2::from_shape(input.fixed_dims::<2>(), input.data::<f32>()).unwrap();
            let in_view =
                in_view.permuted_axes([transpose.perm[0] as usize, transpose.perm[1] as usize]);
            output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
        }
        3 => {
            let in_view =
                ArrayView3::from_shape(input.fixed_dims::<3>(), input.data::<f32>()).unwrap();
            let in_view = in_view.permuted_axes([
                transpose.perm[0] as usize,
                transpose.perm[1] as usize,
                transpose.perm[2] as usize,
            ]);
            output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
        }
        4 => {
            let in_view =
                ArrayView4::from_shape(input.fixed_dims::<4>(), input.data::<f32>()).unwrap();
            let in_view = in_view.permuted_axes([
                transpose.perm[0] as usize,
                transpose.perm[1] as usize,
                transpose.perm[2] as usize,
                transpose.perm[3] as usize,
            ]);
            output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
        }
        _ => todo!("Transpose: Unsupported shape."),
    }
}

fn compute_squeeze(_squeeze: &Squeeze, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::SQUEEZE_IN];
    assert!(input.elem_ty().is_f32());
    let output = &mut outputs[Node::SQUEEZE_OUT];
    output.set_raw_vec(input.data::<f32>().to_vec());
}

fn compute_reduce_mean(rmean: &ReduceMean, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0];
    let output = &mut outputs[0];
    let axes = rmean
        .axes
        .iter()
        .map(|&axis| {
            if axis < 0 {
                (input.dims().len() as i64 + axis) as usize
            } else {
                axis as usize
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(axes.len(), 1);
    let axis = axes[0];
    assert_eq!(input.dims().len(), 3);
    assert_eq!(axis, 2);
    assert!(rmean.keep_dims);
    for i in 0..input.dims()[0] {
        for j in 0..input.dims()[1] {
            let mut sum = 0f32;
            for k in 0..input.dims()[2] {
                sum += input.at_3d(i, j, k);
            }
            *output.at_3d_mut(i, j, 0) = sum / input.dims()[2] as f32;
        }
    }
}

fn compute_loop(_node: &Node, _inputs: &[&Tensor], _outputs: &mut [Tensor]) {
    todo!("loop")
}

fn compute_tile(_node: &Node, _inputs: &[&Tensor], _outputs: &mut [Tensor]) {
    todo!("tile")
}

fn compute_cast(cast: &Cast, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::CAST_IN];
    let output = &mut outputs[Node::CAST_OUT];
    if input.elem_ty().is_i32() && cast.to.is_f32() {
        *output = Tensor::new(
            output.dims().clone(),
            input.data::<i32>().iter().map(|x| *x as f32).collect(),
        );
    } else if input.elem_ty().is_i64() && cast.to.is_i32() {
        *output = Tensor::new(
            output.dims().clone(),
            input.data::<i64>().iter().map(|x| *x as i32).collect(),
        );
    } else {
        todo!()
    }
}

fn compute_batch_normalization(
    batchnorm: &BatchNormalization,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) {
    let data = inputs[Node::BATCHNORM_IN_X];
    let scale = inputs[Node::BATCHNORM_IN_SCALE];
    let bias = inputs[Node::BATCHNORM_IN_B];
    let input_mean = inputs[Node::BATCHNORM_IN_INPUT_MEAN];
    let input_var = inputs[Node::BATCHNORM_IN_INPUT_VAR];
    let output = &mut outputs[Node::BATCHNORM_OUT_Y];

    assert!(!batchnorm.training_mode, "Training mode is not supported.");
    assert!(
        data.elem_ty() == TensorElemType::F32,
        "Input data type must be f32."
    );
    assert!(data.dims().len() == 4, "Input data rank must be 4.");
    assert!(scale.dims().len() == 1, "Scale rank must be 1.");
    assert!(bias.dims().len() == 1, "Bias rank must be 1.");
    assert!(input_mean.dims().len() == 1, "Input mean rank must be 1.");
    assert!(input_var.dims().len() == 1, "Input var rank must be 1.");

    let num_batch = data.dims()[0];
    let num_channel = data.dims()[1];
    let num_dim0 = data.dims()[2];
    let num_dim1 = data.dims()[3];
    let data_strides = data.strides();
    let data_raw = data.data::<f32>();
    let scale_raw = scale.data::<f32>();
    let bias_raw = bias.data::<f32>();
    let input_mean_raw = input_mean.data::<f32>();
    let input_var_raw = input_var.data::<f32>();
    let output_raw = output.data_mut::<f32>();

    for n in 0..num_batch {
        let off_n = n * data_strides[0];
        for b in 0..num_channel {
            let off_b = off_n + b * data_strides[1];
            for h in 0..num_dim0 {
                let off_h = off_b + h * data_strides[2];
                for w in 0..num_dim1 {
                    let off_w = off_h + w; // * data_strides[3];
                    output_raw[off_w] = ((data_raw[off_w] - input_mean_raw[b])
                        / (input_var_raw[b] + batchnorm.epsilon).sqrt())
                        * scale_raw[b]
                        + bias_raw[b];
                }
            }
        }
    }
}

fn compute_slice(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let data = inputs[Node::SLICE_IN_DATA];
    let starts = inputs[Node::SLICE_IN_STARTS].data::<i64>();
    let ends = inputs[Node::SLICE_IN_ENDS].data::<i64>();
    let axes = inputs[Node::SLICE_IN_AXES].data::<i64>();
    let output = &mut outputs[Node::SLICE_OUT];

    assert_eq!(data.dims().len(), 3);

    let axes = axes
        .iter()
        .map(|&axis| {
            if axis < 0 {
                (data.dims().len() as i64 + axis) as usize
            } else {
                axis as usize
            }
        })
        .collect::<Vec<_>>();

    let ones = vec![1i64; axes.len()];
    let steps = inputs
        .get(Node::SLICE_IN_STEPS)
        .map_or(ones.as_slice(), |s| s.data::<i64>());

    assert!(starts.iter().all(|&x| x >= 0));
    assert!(ends.iter().all(|&x| x >= 0));
    assert!(steps.iter().all(|&x| x >= 0));
    assert!(starts.len() == 1, "More than one axis not yet supported.");

    for (((&start, &end), &axis), &step) in starts
        .iter()
        .zip(ends.iter())
        .zip(axes.iter())
        .zip(steps.iter())
    {
        let start = start as usize;
        let end = end as usize;
        let step = step as usize;
        assert_eq!(axis, 2);

        let data = ArrayView3::from_shape(data.fixed_dims::<3>(), data.data::<f32>()).unwrap();
        output.data_mut::<f32>().copy_from_slice(
            data.slice(s![.., .., start..end;step])
                .as_standard_layout()
                .as_slice()
                .unwrap(),
        );
    }
}

fn compute_gather(gather: &Gather, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let data = inputs[0];
    let indices = inputs[1];
    let output = &mut outputs[0];

    assert!(gather.axis >= 0);
    assert!(indices.dims().is_scalar(), "Indices shape: {indices:?}");

    let axis = gather.axis as usize;
    assert_eq!(axis, 1);
    assert_eq!(data.dims().len(), 3);
    assert_eq!(data.dims()[0], 1);

    let gathered = &data.slice_at::<f32>(&[0, indices.data::<i64>()[0] as usize])[..data.dims()[2]];
    output.data_mut().copy_from_slice(gathered);
}

fn compute_reshape(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::RESHAPE_IN];
    let shape = inputs[Node::RESHAPE_SHAPE]
        .data::<i64>()
        .iter()
        .map(|&x| {
            if x == -1 {
                let other_dims_sz: i64 = inputs[Node::RESHAPE_SHAPE]
                    .data::<i64>()
                    .iter()
                    .filter(|x| **x != -1)
                    .product();
                input.dims().total_elems() / other_dims_sz as usize
            } else {
                x as usize
            }
        })
        .collect::<Vec<_>>();
    let output = &mut outputs[Node::RESHAPE_OUT];
    *output = input.clone().reshape_into(shape.into())
}

fn compute_flatten(_flatten: &Flatten, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::FLATTEN_IN];
    let output = &mut outputs[Node::FLATTEN_OUT];
    *output = input.clone().reshape_into(output.dims().clone());
}

// TODO: Better move to another file.
/// Infer `TypedShape`s of output tensors for each node.
/// It skips to infer on nodes without information for inference.
fn infer_shapes(
    model: &Model,
    sorted_nodes: &[NodeId],
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
) {
    let mut values = model.inits.clone();

    for &val_id in &model.inputs {
        let shape = &model.values.inner()[val_id].shape;
        if let Some(shape) = shape {
            let tensor = Tensor::zeros_of_type(shape.elem_ty, shape.dims.clone());
            values.insert(val_id, tensor);
        }
    }

    for &node in sorted_nodes {
        infer_shape(model, &mut values, shapes, node)
    }
}

fn infer_shape(
    model: &Model,
    values: &mut FxHashMap<ValueId, Tensor>,
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    node_id: NodeId,
) {
    let node = &model.nodes[node_id];
    let mut op = node.op.clone();
    let mut inputs = vec![];
    for input in &node.inputs {
        let input = if let Some(input) = values.get(input) {
            input
        } else {
            return;
        };
        inputs.push(input);
    }
    let output_shapes = compute_output_shapes(&mut op, &inputs);
    let mut outputs = vec![];
    for shape in &output_shapes {
        outputs.push(Tensor::zeros_of_type(shape.elem_ty, shape.dims.clone()));
    }
    for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
        values.insert(val, output);
    }
    shapes.insert(node_id, (op, output_shapes));
}
