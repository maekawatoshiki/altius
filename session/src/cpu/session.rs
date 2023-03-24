#[cfg(feature = "cblas")]
use super::gemm::sgemm2;
use super::{
    conv2d::{self, Conv2dCtx},
    fast_math::{fast_gelu, fast_sigmoid},
    gemm::sgemm,
    thread::ThreadCtx,
};

use crate::{cpu::fast_math::fast_sum_exp, NodeExecutionPlan, SessionError};
use altius_core::{
    model::Model,
    node::{Node, NodeId},
    op::{
        BatchNormalization, Cast, Concat, Flatten, Gather, Gemm, HardSigmoid, LayerNormalization,
        LeakyReLU, MaxPool, Op, ReduceMax, ReduceMean, Resize, Softmax, Split, Squeeze, Transpose,
        Unsqueeze,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
    value::ValueId,
};
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;
#[cfg(feature = "x64-fusion")]
use dynasm::dynasm;
#[cfg(feature = "x64-fusion")]
use dynasmrt::{x64::Assembler, DynasmApi};
use ndarray::{s, ArrayView, ArrayView3, Axis, Dim, Ix};
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;

use std::{
    cell::RefCell,
    simd::{Simd, SimdFloat, StdFloat},
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
pub(super) use cuda::*;

pub struct CPUSession {
    pub(super) model: Model,
    #[cfg(feature = "cuda")]
    pub(super) cudnn_ctx: SafeCudnnContext,
    pub(super) execution_plans: Vec<NodeExecutionPlan>,
    pub(super) inferred_shapes: FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    pub(super) enable_profiling: bool,
    pub(super) values: ThreadLocal<RefCell<FxHashMap<ValueId, Tensor>>>,
    pub(super) dummy_value: Tensor,
    pub(super) tctx: ThreadCtx,
    #[cfg(feature = "x64-fusion")]
    asm_ops: Assmembler,
}

// TODO: Snippets for x64 codegen.
//
// #[cfg(feature = "x64-fusion")]
// let mut ops = Assembler::new().unwrap();
//
// #[cfg(feature = "x64-fusion")]
// let entry = ops.offset();
// #[cfg(feature = "x64-fusion")]
// dynasm!(ops
//     // rdi = input.0 addr (*const f32)
//     // rsi = input.0 addr (*const f32)
//     // rdx = output.0 addr (*mut f32)
//     // rcx = len (u64)
//     ; .arch x64
//     ; vmovups ymm0, [rdi]
//     ; vmovups ymm1, [rsi]
//     ; vaddps ymm0, ymm0, ymm1
//     ; vmovups [rdx], ymm0
//     ; ret
// );
// #[cfg(feature = "x64-fusion")]
// let buf = ops.finalize().unwrap();
// #[cfg(feature = "x64-fusion")]
// let entry_fn: extern "C" fn(*const f32, *const f32, *mut f32, u64) -> f32 =
//     unsafe { std::mem::transmute(buf.ptr(entry)) };

impl CPUSession {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn run(&self, inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        #[cfg(not(target_arch = "wasm32"))]
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

        #[cfg(not(feature = "heavy-log"))]
        for node in &self.execution_plans {
            self.run_node(&mut profile, values, node.node_id)?;

            for val in &node.free_vals {
                values.get_mut(val).unwrap().set_raw_vec::<u8>(Vec::new())
            }
        }

        #[cfg(feature = "heavy-log")]
        for (i, node) in self.execution_plans.iter().enumerate() {
            let start = Instant::now();

            self.run_node(&mut profile, values, node.node_id);

            log::info!(
                "{}/{} {}({}) {:?}",
                i,
                self.execution_plans.len(),
                self.model.nodes[node.node_id].op.name(),
                self.model.nodes[node.node_id]
                    .name
                    .as_ref()
                    .unwrap_or(&"".to_string()),
                start.elapsed()
            );

            for val in &node.free_vals {
                values.get_mut(val).unwrap().set_raw_vec::<u8>(Vec::new())
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
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
    ) -> Result<(), SessionError> {
        let node = &self.model.nodes[node_id];
        let inputs = node
            .inputs
            .iter()
            .map(|input| values.get(input).unwrap_or(&self.dummy_value))
            .collect::<Vec<_>>();
        // Use inferred shapes if any.
        let (op, output_shapes) = self.inferred_shapes.get(&node_id).cloned().map_or_else(
            || {
                let mut op = node.op.clone();
                let output_shapes =
                    op.compute_output_shapes(&inputs, node.outputs.len(), self.model.opset_version);
                output_shapes.map(|os| (op, os))
            },
            |result| Ok(result),
        )?;
        let mut outputs = output_shapes
            .into_iter()
            .map(|TypedShape { elem_ty, dims }| Tensor::uninit_of_type(elem_ty, dims))
            .collect::<Vec<_>>();

        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();

        // Actual kernel runs here.
        match op {
            Op::Conv2d(ref conv) => conv2d::compute(&mut Conv2dCtx {
                #[cfg(feature = "cuda")]
                cudnn: &self.cudnn_ctx,
                op: conv,
                inputs: &inputs,
                outputs: &mut outputs,
                tctx: &self.tctx,
            }),
            Op::Add => compute_add(&self.tctx, &inputs, &mut outputs),
            Op::Sub => compute_sub(&self.tctx, &inputs, &mut outputs),
            Op::Mul => compute_mul(&self.tctx, &inputs, &mut outputs),
            Op::Div => compute_div(&self.tctx, &inputs, &mut outputs),
            Op::Greater => compute_greater(node, &inputs, &mut outputs),
            Op::Pow => compute_pow(node, &inputs, &mut outputs),
            Op::Sqrt => compute_sqrt(node, &inputs, &mut outputs),
            Op::MaxPool(ref maxpool) => compute_max_pool(maxpool, &inputs, &mut outputs),
            Op::GlobalAveragePool => compute_gavg_pool(node, &inputs, &mut outputs)?,
            Op::Expand => compute_expand(&inputs, &mut outputs)?,
            Op::Reshape => compute_reshape(node, &inputs, &mut outputs),
            Op::Flatten(ref flatten) => compute_flatten(flatten, &inputs, &mut outputs),
            Op::MatMul => compute_mat_mul(node, &inputs, &mut outputs),
            Op::Gemm(ref gemm) => compute_gemm(gemm, &inputs, &mut outputs),
            Op::ReLU => compute_relu(node, &inputs, &mut outputs),
            Op::HardSigmoid(ref hs) => compute_hard_sigmoid(hs, &inputs, &mut outputs),
            Op::LeakyReLU(ref leaky) => compute_leaky_relu(leaky, &inputs, &mut outputs),
            Op::Gelu => compute_gelu(&self.tctx, &inputs, &mut outputs),
            Op::Sigmoid => compute_sigmoid(&self.tctx, &inputs, &mut outputs),
            Op::Erf => compute_erf(&inputs, &mut outputs),
            Op::Tanh => compute_tanh(&inputs, &mut outputs),
            Op::Clip => return Err(SessionError::Message("Clip: Kernel not implemented".into())),
            Op::Where => compute_where(&inputs, &mut outputs),
            Op::Softmax(ref softmax) => compute_softmax(&self.tctx, softmax, &inputs, &mut outputs),
            Op::Resize(ref resize) => compute_resize(&self.tctx, resize, &inputs, &mut outputs),
            Op::Concat(ref concat) => compute_concat(concat, &inputs, &mut outputs)?,
            Op::Transpose(ref trans) => compute_transpose(trans, &inputs, &mut outputs),
            Op::Squeeze(ref squeeze) => compute_squeeze(squeeze, &inputs, &mut outputs),
            Op::Unsqueeze(ref unsqueeze) => compute_unsqueeze(unsqueeze, &inputs, &mut outputs),
            Op::ReduceMin(_) => {
                return Err(SessionError::Message(
                    "ReduceMin: Kernel not implemented".into(),
                ))
            }
            Op::ReduceMax(ref rmax) => compute_reduce_max(rmax, &inputs, &mut outputs),
            Op::ReduceMean(ref rmean) => compute_reduce_mean(rmean, &inputs, &mut outputs),
            Op::Round => {
                return Err(SessionError::Message(
                    "Round: Kernel not implemented".into(),
                ))
            }
            Op::Exp => return Err(SessionError::Message("Exp: Kernel not implemented".into())),
            Op::Loop => compute_loop(node, &inputs, &mut outputs),
            Op::Tile => compute_tile(node, &inputs, &mut outputs),
            Op::Cast(ref cast) => compute_cast(cast, &inputs, &mut outputs),
            Op::BatchNormalization(ref batchnorm) => {
                compute_batch_normalization(batchnorm, &inputs, &mut outputs)
            }
            Op::LayerNormalization(ref ln) => {
                compute_layer_normalization(&self.tctx, ln, &inputs, &mut outputs)
            }
            Op::Split(ref split) => {
                compute_split(self.model.opset_version, split, &inputs, &mut outputs)
            }
            Op::Slice => compute_slice(node, &inputs, &mut outputs),
            Op::Gather(ref gather) => compute_gather(gather, &inputs, &mut outputs),
            Op::Shape(_) => {
                return Err(SessionError::Message(
                    "Shape: Kernel not implemented".into(),
                ))
            }
            Op::NonMaxSuppression => {
                return Err(SessionError::Message(
                    "NonMaxSuppression: Kernel not implemented".into(),
                ))
            }
            Op::Constant(_) => {
                return Err(SessionError::Message(
                    "Constant: Kernel not implemented".into(),
                ))
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.enable_profiling {
            let elapsed = start.elapsed();
            *profile.entry(op.name()).or_insert(Duration::ZERO) += elapsed;
        }

        for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
            values.insert(val, output);
        }

        Ok(())
    }
}

fn compute_gavg_pool(
    _node: &Node,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) -> Result<(), SessionError> {
    let input = inputs[Op::GLOBALAVERAGEPOOL_IN];
    let output = &mut outputs[Op::GLOBALAVERAGEPOOL_OUT];

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    let Some(&[_, _, h, w]) = input.dims().get(0..4) else {
        return Err(SessionError::Message("Input must be four dimensions".into()))
    };
    let Some([isn, isc, _, _]) = input.strides().get(0..4) else { panic!() };
    let area = h * w;
    let osn = output.strides()[0];
    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();

    for (o_channels, i_channels) in output.chunks_mut(osn).zip(input.chunks(*isn)) {
        for (o_channel, i_channel) in o_channels.iter_mut().zip(i_channels.chunks(*isc)) {
            let sum: f32 = i_channel.iter().sum();
            *o_channel = sum / area as f32;
        }
    }

    Ok(())
}

fn compute_expand(inputs: &[&Tensor], outputs: &mut [Tensor]) -> Result<(), SessionError> {
    let input = inputs[0];
    // let shape = inputs[1]
    //     .data::<i64>()
    //     .iter()
    //     .map(|&x| x as usize)
    //     .collect::<Vec<_>>();
    let output = &mut outputs[0];

    assert!(input.dims().len() == 4);
    assert!(input.dims()[0..3] == [1, 1, 1]);
    assert!(output.dims().len() == 4);
    assert!(input.dims()[0..2] == [1, 1]);
    assert!(input.elem_ty().is_i64());

    let chunk = input.dims()[3];
    let input = input.data::<i64>();
    let output = output.data_mut::<i64>();

    output.chunks_mut(chunk).for_each(|o| {
        o.copy_from_slice(input);
    });

    Ok(())
}

fn compute_max_pool(maxpool: &MaxPool, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Op::MAXPOOL_IN];
    let output = &mut outputs[Op::MAXPOOL_OUT];

    let kernel = &maxpool.kernel_shape;
    let stride = &maxpool.strides;

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    let padding = &maxpool.padding;
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

    let pad_t = padding[0] as isize;
    let pad_l = padding[1] as isize;

    for _ in 0..outer {
        let mut y = -pad_t;
        for ay in 0..output_h {
            let mut x = -pad_l;
            let output = &mut output[ay * output_w..];
            let fy_min = (-y).max(0) as usize;
            let fy_max = kernel_h.min((input_h as isize - y) as usize);
            for out in output.iter_mut().take(output_w) {
                let mut max = f32::MIN;
                let fx_min = (-x).max(0) as usize;
                let fx_max = kernel_w.min((input_w as isize - x) as usize);
                for fy in fy_min..fy_max {
                    let oy = y + fy as isize;
                    for fx in fx_min..fx_max {
                        let ox = x + fx as isize;
                        max = input[oy as usize * input_w + ox as usize].max(max);
                    }
                }
                *out = if max == f32::MIN { 0.0 } else { max };
                x += stride_w as isize
            }
            y += stride_h as isize
        }
        input = &input[input_hw..];
        output = &mut output[output_hw..];
    }
}

macro_rules! op_bin_elemwise {
    ($name:ident, $op:tt) => { paste::item! {
        fn $name(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
            let input_a = inputs[0];
            let input_b = inputs[1];
            let output = &mut outputs[0];

            let adims = input_a.dims();
            let bdims = input_b.dims();

            if adims == bdims {
                let input_a = input_a.data::<f32>();
                let input_b = input_b.data::<f32>();
                let output = output.data_mut::<f32>();
                let chunk = 100000;

                tctx.scope(|scope| {
                    input_a
                        .chunks(chunk)
                        .zip(input_b.chunks(chunk))
                        .zip(output.chunks_mut(chunk))
                        .for_each(|((input_a, input_b), output)| {
                            scope.spawn(move || {
                                for ((a, b), o) in
                                    input_a.iter().zip(input_b.iter()).zip(output.iter_mut())
                                {
                                    // Auto-vectorized by LLVM
                                    *o = a $op b;
                                }
                            });
                        });
                });

                return;
            }

            if bdims.is_scalar() {
                let b = input_b.data::<f32>()[0];
                let output = output.data_mut::<f32>();

                for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
                    *o = a $op b;
                }

                return;
            }

            if adims.is_scalar() {
                let a = input_a.data::<f32>()[0];
                let output = output.data_mut::<f32>();

                for (b, o) in input_b.data::<f32>().iter().zip(output.iter_mut()) {
                    *o = a $op b;
                }

                return;
            }

            [< $name _general >](tctx, input_a, input_b, output);
        }

        fn [< $name _general >](_tctx: &ThreadCtx, input_a: &Tensor, input_b: &Tensor, output: &mut Tensor) {
            fn compute(a_stride: &[usize],
                    b_stride: &[usize],
                    o_stride: &[usize],
                    o_shape: &[usize],
                    mut a: &[f32],
                    mut b: &[f32],
                    o: &mut [f32]) {
                if a_stride.len() == 1 {
                    let len = o_shape[0];
                    let a_stride = a_stride[0];
                    let b_stride = b_stride[0];

                    if a_stride == 1 && b_stride == 1 {
                        for (o, (a, b)) in o[..len].iter_mut().zip(a[..len].iter().zip(b[..len].iter())) {
                            *o = a $op b;
                        }
                        return;
                    }

                    if a_stride == 0 && b_stride == 0 {
                        let a = a[0];
                        let b = b[0];
                        for o in &mut o[..len] {
                            *o = a $op b;
                        }
                        return;
                    }

                    if a_stride == 1 && b_stride == 0 {
                        let b = b[0];
                        for (o, a) in o[..len].iter_mut().zip(a[..len].iter()) {
                            *o = a $op b;
                        }
                        return;
                    }

                    if a_stride == 0 && b_stride == 1 {
                        let a = a[0];
                        for (o, b) in o[..len].iter_mut().zip(b[..len].iter()) {
                            *o = a $op b;
                        }
                        return;
                    }

                    for o in &mut o[..len] {
                        *o = a[0] $op b[0];
                        a = &a[a_stride..];
                        b = &b[b_stride..];
                    }
                    return;
                }

                for i in 0..o_shape[0] {
                    compute(&a_stride[1..],
                            &b_stride[1..],
                            &o_stride[1..],
                            &o_shape[1..],
                            &a[i * a_stride[0]..],
                            &b[i * b_stride[0]..],
                            &mut o[i * o_stride[0]..]);
                }
            }

            let o_shape = output.dims().clone();
            let o_stride = output.strides().to_vec();
            let a_stride = input_a
                .strides_for_broadcasting(&o_shape)
                .unwrap();
            let b_stride  = input_b
                .strides_for_broadcasting(&o_shape)
                .unwrap();
            let input_a = input_a.data::<f32>();
            let input_b = input_b.data::<f32>();
            let output = output.data_mut::<f32>();

            assert!(o_shape.len() > 1);

            compute(&a_stride, &b_stride, &o_stride, &o_shape, &input_a, &input_b, output);
        }
    }};
}

op_bin_elemwise!(compute_add, +);
op_bin_elemwise!(compute_sub, -);
op_bin_elemwise!(compute_mul, *);
op_bin_elemwise!(compute_div, /);

fn compute_greater(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_0 = inputs[0];
    let input_1 = inputs[1];
    let output = &mut outputs[0];

    if input_1.dims().is_scalar() {
        let y = input_1.data::<f32>()[0];
        for (&x, o) in input_0
            .data::<f32>()
            .iter()
            .zip(output.data_mut::<bool>().iter_mut())
        {
            *o = x > y;
        }
        return;
    }

    todo!(
        "A shape: {:?}, B shape: {:?}",
        input_0.dims(),
        input_1.dims()
    )
}

fn compute_pow(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[0];
    let input_b = inputs[1];
    let output = outputs[0].data_mut();

    if input_a.dims() == input_b.dims() {
        for ((a, b), o) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .zip(output.iter_mut())
        {
            *o = a.powf(*b);
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
    let input_a = inputs[Op::MATMUL_IN_A];
    let input_b = inputs[Op::MATMUL_IN_B];
    let output = &mut outputs[Op::MATMUL_OUT];

    let adim = input_a.dims();
    let bdim = input_b.dims();

    assert!(
        (adim.len() == 2 && bdim.len() == 2 && adim[1] == bdim[0])
            || (adim.len() == 3 && bdim.len() == 2 && adim[2] == bdim[0])
            || (adim.len() == 3 && bdim.len() == 3 && adim[0] == bdim[0] && adim[2] == bdim[1])
            || (adim.len() == 4
                && bdim.len() == 4
                && adim[0] == 1
                && bdim[0] == 1
                && adim[1] == bdim[1]),
        "A shape: {adim:?}, B shape: {bdim:?}"
    );

    if adim.len() == 4 && bdim.len() == 4 {
        let [_one, _batch, m, _k] = input_a.fixed_dims::<4>();
        let [_one, _batch, k, n] = input_b.fixed_dims::<4>();
        // TODO: Are there a better way?
        output
            .data_mut::<f32>()
            .chunks_mut(m * n)
            .zip(input_a.data::<f32>().chunks(m * k))
            .zip(input_b.data::<f32>().chunks(k * n))
            .for_each(|((c, a), b)| {
                sgemm(m, k, n, 1., a, k, b, n, 0., c, n);
            });
    } else if adim.len() == 3 && bdim.len() == 2 {
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
    let input_a = inputs[Op::GEMM_IN_A];
    let input_b = inputs[Op::GEMM_IN_B];
    let input_c = inputs[Op::GEMM_IN_C];
    let output = &mut outputs[Op::GEMM_OUT];

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
    let input: &[f32] = inputs[Op::RELU_IN].data();
    let output: &mut [f32] = outputs[Op::RELU_OUT].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = i.max(0.0);
    }
}

fn compute_hard_sigmoid(hs: &HardSigmoid, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[Op::HARDSIGMOID_IN].data();
    let output: &mut [f32] = outputs[Op::HARDSIGMOID_OUT].data_mut();

    for (&i, o) in input.iter().zip(output.iter_mut()) {
        *o = hs.alpha.mul_add(i, hs.beta).clamp(0.0, 1.0)
    }
}

fn compute_leaky_relu(leaky: &LeakyReLU, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = if *i < 0.0 { leaky.alpha * i } else { *i };
    }
}

fn compute_gelu(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();
    let n = tctx.num_threads();

    tctx.scope(|scope| {
        input
            .chunks(input.len() / n)
            .zip(output.chunks_mut(input.len() / n))
            .for_each(|(input, output)| {
                scope.spawn(move || {
                    fast_gelu(output, input);
                })
            });
    });
}

fn tanh(mut data: &mut [f32]) {
    const LOWER_RANGE: f32 = -9f32;
    const UPPER_RANGE: f32 = 9f32;
    const ALPHA_13: f32 = -2.76076847742355e-16f32;
    const ALPHA_11: f32 = 2.00018790482477e-13f32;
    const ALPHA_9: f32 = -8.60467152213735e-11f32;
    const ALPHA_7: f32 = 5.12229709037114e-08f32;
    const ALPHA_5: f32 = 1.48572235717979e-05f32;
    const ALPHA_3: f32 = 6.37261928875436e-04f32;
    const ALPHA_1: f32 = 4.89352455891786e-03f32;
    const BETA_6: f32 = 1.19825839466702e-06f32;
    const BETA_4: f32 = 1.18534705686654e-04f32;
    const BETA_2: f32 = 2.26843463243900e-03f32;
    const BETA_0: f32 = 4.89352518554385e-03f32;

    const SIMD_LEN: usize = 8;

    let mut len = data.len();

    while len >= SIMD_LEN {
        let vals = Simd::<f32, SIMD_LEN>::from_slice(data);
        let vals = vals.simd_clamp(Simd::splat(LOWER_RANGE), Simd::splat(UPPER_RANGE));

        let vals_squared = vals * vals;

        let p = vals_squared.mul_add(Simd::splat(ALPHA_13), Simd::splat(ALPHA_11));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_9));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_7));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_5));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_3));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_1));
        let p = p * vals;

        let q = vals_squared.mul_add(Simd::splat(BETA_6), Simd::splat(BETA_4));
        let q = q.mul_add(vals_squared, Simd::splat(BETA_2));
        let q = q.mul_add(vals_squared, Simd::splat(BETA_0));

        data[0..SIMD_LEN].copy_from_slice((p / q).as_ref());

        len -= SIMD_LEN;
        data = &mut data[SIMD_LEN..];
    }

    for x in data {
        let val = *x;
        let val = val.clamp(LOWER_RANGE, UPPER_RANGE);

        let val_squared = val * val;

        let p = val_squared.mul_add(ALPHA_13, ALPHA_11);
        let p = p.mul_add(val_squared, ALPHA_9);
        let p = p.mul_add(val_squared, ALPHA_7);
        let p = p.mul_add(val_squared, ALPHA_5);
        let p = p.mul_add(val_squared, ALPHA_3);
        let p = p.mul_add(val_squared, ALPHA_1);
        let p = p * val;

        let q = val_squared.mul_add(BETA_6, BETA_4);
        let q = q.mul_add(val_squared, BETA_2);
        let q = q.mul_add(val_squared, BETA_0);

        *x = p / q
    }
}

fn compute_sigmoid(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[Op::SIGMOID_IN].data();
    let output: &mut [f32] = outputs[Op::SIGMOID_OUT].data_mut();

    let threshold = 512;
    let chunk_size = output.len() / tctx.num_threads();

    if chunk_size > threshold {
        tctx.scope(|scope| {
            input
                .chunks(chunk_size)
                .zip(output.chunks_mut(chunk_size))
                .for_each(|(input, output)| scope.spawn(move || fast_sigmoid(output, input)))
        })
    } else {
        fast_sigmoid(output, input)
    }
}

fn compute_erf(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();

    for (&i, o) in input.iter().zip(output.iter_mut()) {
        *o = fastapprox::faster::erf(i);
    }
}

fn compute_tanh(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0].data::<f32>();
    let output = outputs[0].data_mut::<f32>();
    output.copy_from_slice(input);
    tanh(output);
}

fn compute_where(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let condition = inputs[0];
    let x = inputs[1];
    let y = inputs[2];
    let output = &mut outputs[0];

    // TODO: We should do better handling of broadcasting.
    if condition.dims() == y.dims() && x.dims().is_scalar() {
        let output = output.data_mut::<f32>();
        let x = x.data::<f32>()[0];
        let y = y.data::<f32>();
        let condition = condition.data::<bool>();
        for (&c, (&y, o)) in condition.iter().zip(y.iter().zip(output.iter_mut())) {
            *o = if c { x } else { y }
        }
        return;
    }

    assert!(y.dims().is_scalar());
    assert!(
        x.dims().len() == 4
            && condition.dims().len() == 4
            && condition.dims()[0] == 1
            && condition.dims()[1] == 1
    );

    let out0 = output.dims()[0];
    let out1 = output.dims()[1];
    let out2 = output.dims()[2];
    let out3 = output.dims()[3];
    let mut x = x.data::<f32>();
    let y = y.data::<f32>()[0];
    let condition = condition.data::<bool>();
    let mut output = output.data_mut::<f32>();

    for _o0 in 0..out0 {
        for _o1 in 0..out1 {
            for ((&c, &x), o) in condition
                .iter()
                .zip(x[..out2 * out3].iter())
                .zip(output[..out2 * out3].iter_mut())
            {
                *o = if c { x } else { y };
            }
            x = &x[out2 * out3..];
            output = &mut output[out2 * out3..];
        }
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

    assert!(softmax.axis == -1 || softmax.axis == (input.dims().len() - 1) as i64);

    let axis_len = *input.dims().last().unwrap();
    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();

    let batch = (output.len() / tctx.num_threads() / axis_len).max(1);

    tctx.scope(|scope| {
        output
            .chunks_mut(axis_len * batch)
            .zip(input.chunks(axis_len * batch))
            .for_each(|(output, input)| {
                scope.spawn(move || {
                    output
                        .chunks_mut(axis_len)
                        .zip(input.chunks(axis_len))
                        .for_each(|(output, input)| {
                            let sum = fast_sum_exp(output, input);
                            let recip_sum = 1. / sum;
                            for o in output {
                                *o *= recip_sum;
                            }
                        });
                })
            });
    });
}

fn fast_sum(mut slice: &[f32]) -> f32 {
    const SIMD_LEN: usize = 8;
    let mut sum = Simd::<f32, SIMD_LEN>::splat(0f32);
    let mut len = slice.len();

    while len >= SIMD_LEN {
        sum += Simd::<f32, SIMD_LEN>::from_slice(slice);
        slice = &slice[SIMD_LEN..];
        len -= SIMD_LEN
    }

    sum.reduce_sum() + slice.iter().sum::<f32>()
}

fn fast_sum_squares(mut slice: &[f32]) -> f32 {
    const SIMD_LEN: usize = 8;
    let mut sum = Simd::<f32, SIMD_LEN>::splat(0f32);
    let mut len = slice.len();

    while len >= SIMD_LEN {
        let lane = Simd::<f32, SIMD_LEN>::from_slice(slice);
        sum += lane * lane;
        slice = &slice[SIMD_LEN..];
        len -= SIMD_LEN
    }

    sum.reduce_sum() + slice.iter().map(|x| x * x).sum::<f32>()
}

fn compute_resize(tctx: &ThreadCtx, resize: &Resize, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    assert!(matches!(inputs.len(), 3 | 4));

    let input = inputs[Op::RESIZE_IN_X];
    // let sizes = inputs[Op::RESIZE_IN_SIZES];
    let output = &mut outputs[Op::RESIZE_OUT];

    let batch_size = input.dims()[0];
    let input_c = input.dims()[1];
    let input_h = input.dims()[2];
    let input_w = input.dims()[3];
    let output_h = output.dims()[2];
    let output_w = output.dims()[3];
    let input_hw = input_h * input_w;
    let outer = batch_size * input_c;

    let scale = output_h as f32 / input_h as f32;

    if resize.mode == "nearest" {
        let mut input = input.data::<f32>();
        let mut output = output.data_mut::<f32>();

        for _ in 0..outer {
            for (h, o) in (0..output_h).zip(output.chunks_mut(output_w)) {
                let ih = (h as f32 / scale) as usize;
                let ihw = input_w * ih;
                for (w, o) in (0..output_w).zip(o.iter_mut()) {
                    let iw = (w as f32 / scale) as usize;
                    *o = input[ihw + iw];
                }
            }
            output = &mut output[output_h * output_w..];
            input = &input[input_hw..];
        }
    } else {
        log::info!("Resize: Current implementation uses bilinear interpolation!");

        // TODO: Multi-threading could make the performance worse depending on height and/or width.
        tctx.scope(|scope| {
            let mut input = input.data::<f32>();
            let mut output = output.data_mut::<f32>();

            for _ in 0..outer {
                for (h, o) in (0..output_h).zip(output.chunks_mut(output_w)) {
                    let ihf = (h as f32 / scale - 0.5).max(0.);
                    let ih = ihf as usize;
                    let ih0 = ih.min(input_h - 1);
                    let ih1 = (ih + 1).min(input_h - 1);
                    let ih0w = input_w * ih0;
                    let ih1w = input_w * ih1;
                    scope.spawn(move || {
                        for (w, o) in (0..output_w).zip(o.iter_mut()) {
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

                            *o = r
                        }
                    });
                }
                output = &mut output[output_h * output_w..];
                input = &input[input_hw..];
            }
        });
    }
}

fn compute_concat(
    concat: &Concat,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) -> Result<(), SessionError> {
    // TODO: Stop using ndarray!
    macro_rules! concat {
        ($n:expr) => {{
            let mut views = vec![];
            for input in inputs {
                views.push(
                    ArrayView::<f32, Dim<[Ix; $n]>>::from_shape(
                        input.fixed_dims::<$n>(),
                        input.data::<f32>(),
                    )
                    .unwrap(),
                );
            }
            ndarray::concatenate(Axis(concat.axis as usize), &views)
                .unwrap()
                .as_standard_layout()
                .to_owned()
                .into_raw_vec()
        }};
    }

    let output = &mut outputs[Op::CONCAT_OUT];
    let d = output.dims().len();

    output.set_raw_vec(match d {
        1 => concat!(1),
        2 => concat!(2),
        3 => concat!(3),
        4 => concat!(4),
        5 => concat!(5),
        d => {
            return Err(SessionError::Message(
                format!("Transpose: {d} dimensions not supported yet").into(),
            ))
        }
    });

    Ok(())
}

fn compute_transpose(transpose: &Transpose, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    struct PermIter {
        num_axes: usize,
        index: Vec<usize>,
        upper_bound: Vec<usize>,
        stride: Vec<usize>,
    }

    #[inline(always)]
    fn next_index(perm: &mut PermIter, mut index: usize) -> usize {
        let mut pos = perm.num_axes - 1;
        index += perm.stride[pos];
        perm.index[pos] += 1;
        if perm.index[pos] < perm.upper_bound[pos] {
            return index;
        }

        index -= perm.stride[pos] * perm.index[pos];
        perm.index[pos] = 0;
        if pos == 0 {
            return index;
        }

        loop {
            pos -= 1;
            index += perm.stride[pos];
            perm.index[pos] += 1;
            if perm.index[pos] < perm.upper_bound[pos] {
                break;
            }
            index -= perm.stride[pos] * perm.index[pos];
            perm.index[pos] = 0;
            if pos == 0 {
                break;
            }
        }
        index
    }

    let input = inputs[Op::TRANSPOSE_IN];
    let output = &mut outputs[Op::TRANSPOSE_OUT];
    assert!(input.elem_ty().is_f32());

    let in_dims = input.dims();
    let in_strides = input.strides();
    let out_dims = output.dims();
    let num_axes = in_dims.len();
    let new_strides = transpose
        .perm
        .iter()
        .map(|&axis| in_strides[axis as usize])
        .collect::<Vec<_>>();

    let mut num_blocks = 1;
    let mut num_elems_in_block = 1;
    let mut suffix = true;
    let mut reduced_num_axes = 0;

    for i in (0..num_axes).rev() {
        let input_axis = transpose.perm[i] as usize;
        if suffix && input_axis == i {
            num_elems_in_block *= in_dims[input_axis];
        } else {
            suffix = false;
            num_blocks *= in_dims[input_axis];
            reduced_num_axes += 1;
        }
    }

    let mut perm = PermIter {
        num_axes: reduced_num_axes,
        index: vec![0; reduced_num_axes],
        upper_bound: out_dims[0..reduced_num_axes].to_vec(),
        stride: new_strides[0..reduced_num_axes].to_vec(),
    };

    let source = input.data::<f32>();
    let target = output.data_mut::<f32>();
    let mut src_idx = 0;

    if num_elems_in_block == 1 {
        for i in 0..num_blocks {
            unsafe { *target.get_unchecked_mut(i) = *source.get_unchecked(src_idx) };
            src_idx = next_index(&mut perm, src_idx);
        }
    } else if num_blocks == 1 {
        target.copy_from_slice(source);
    } else {
        for i in 0..num_blocks {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(src_idx),
                    target.as_mut_ptr().add(i * num_elems_in_block),
                    num_elems_in_block,
                )
            };
            src_idx = next_index(&mut perm, src_idx);
        }
    }
}

fn compute_squeeze(_squeeze: &Squeeze, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Op::SQUEEZE_IN];
    let output = &mut outputs[Op::SQUEEZE_OUT];
    output.copy_data_from(input);
}

fn compute_unsqueeze(_: &Unsqueeze, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0];
    let output = &mut outputs[0];
    output.copy_data_from(input);
}

fn compute_reduce_max(rmax: &ReduceMax, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0];
    let output = &mut outputs[0];
    assert!(rmax.axes.is_empty());
    assert!(!rmax.keep_dims);

    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();

    output[0] = *input
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
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

    let axis_len = input.dims()[2];
    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();
    let r_axis_len = 1.0 / axis_len as f32;

    input
        .chunks(axis_len)
        .zip(output.iter_mut())
        .for_each(|(input, output)| {
            let sum = fast_sum(input);
            *output = sum * r_axis_len;
        });
}

fn compute_loop(_node: &Node, _inputs: &[&Tensor], _outputs: &mut [Tensor]) {
    todo!("loop")
}

fn compute_tile(_node: &Node, _inputs: &[&Tensor], _outputs: &mut [Tensor]) {
    todo!("tile")
}

fn compute_cast(cast: &Cast, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Op::CAST_IN];
    let output = &mut outputs[Op::CAST_OUT];
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
    } else if input.elem_ty().is_i64() && cast.to.is_f32() {
        let input = input.data::<i64>();
        let output = output.data_mut::<f32>();
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = *i as f32;
        }
    } else if input.elem_ty().is_f32() && cast.to.is_bool() {
        let input = input.data::<f32>();
        let output = output.data_mut::<bool>();
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = *i != 0.0;
        }
    } else {
        todo!("cast {:?} -> {:?}", input.elem_ty(), cast.to)
    }
}

fn compute_batch_normalization(
    batchnorm: &BatchNormalization,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) {
    let data = inputs[Op::BATCHNORM_IN_X];
    let scale = inputs[Op::BATCHNORM_IN_SCALE];
    let bias = inputs[Op::BATCHNORM_IN_B];
    let input_mean = inputs[Op::BATCHNORM_IN_INPUT_MEAN];
    let input_var = inputs[Op::BATCHNORM_IN_INPUT_VAR];
    let output = &mut outputs[Op::BATCHNORM_OUT_Y];

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

fn compute_layer_normalization(
    tctx: &ThreadCtx,
    ln: &LayerNormalization,
    inputs: &[&Tensor],
    outputs: &mut [Tensor],
) {
    let data = inputs[0];
    let scale = inputs[1].data::<f32>();
    let bias = inputs[2].data::<f32>();
    let output = &mut outputs[0];

    assert!(
        ln.axis == -1 || ln.axis == *data.dims().last().unwrap() as i64,
        "Axis must be the last dimension."
    );
    assert!(ln.stash_type == 1, "Stash type must be 1.");
    assert!(
        data.elem_ty() == TensorElemType::F32,
        "Input data type must be f32."
    );

    let axis_len = *data.dims().last().unwrap();
    let data = data.data::<f32>();
    let output = output.data_mut::<f32>();

    let batch = (data.len() / tctx.num_threads() / axis_len).max(1);

    tctx.scope(|scope| {
        data.chunks(axis_len * batch)
            .zip(output.chunks_mut(axis_len * batch))
            .for_each(|(data, output)| {
                scope.spawn(move || {
                    data.chunks(axis_len)
                        .zip(output.chunks_mut(axis_len))
                        .for_each(|(input, output)| {
                            let inv_axis_len = (axis_len as f32).recip();
                            let mean = fast_sum(input) * inv_axis_len;
                            for (&i, o) in input.iter().zip(output.iter_mut()) {
                                *o = i - mean;
                            }
                            let inv_mean = fast_sum_squares(output)
                                .mul_add(inv_axis_len, ln.epsilon)
                                .sqrt()
                                .recip();
                            for ((&scale, &bias), o) in
                                scale.iter().zip(bias.iter()).zip(output.iter_mut())
                            {
                                *o = (*o * inv_mean).mul_add(scale, bias)
                            }
                        });
                });
            });
    });
}

fn compute_split(opset_version: i64, split: &Split, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    assert!(split.axis >= 0);
    let axis = split.axis as usize;
    assert_eq!(axis, inputs[0].dims().len() - 1);
    let axis_len = *inputs[0].dims().as_slice().last().unwrap();
    let input = inputs[0].data::<f32>();

    let split = if opset_version >= 13 {
        inputs[1].data::<i64>()
    } else {
        &split.split
    };

    let mut offset = vec![0; split.len()];
    for input in input.chunks(axis_len) {
        let mut s = 0;
        for (i, (sp, output)) in split.iter().zip(outputs.iter_mut()).enumerate() {
            let input = &input[s..s + *sp as usize];
            let output = output.data_mut::<f32>();
            output[offset[i]..offset[i] + input.len()].copy_from_slice(input);
            offset[i] += *sp as usize;
            s += *sp as usize
        }
    }
}

fn compute_slice(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let data = inputs[Op::SLICE_IN_DATA];
    let starts = inputs[Op::SLICE_IN_STARTS].data::<i64>();
    let ends = inputs[Op::SLICE_IN_ENDS].data::<i64>();
    let axes = inputs[Op::SLICE_IN_AXES].data::<i64>();
    let output = &mut outputs[Op::SLICE_OUT];

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
        .get(Op::SLICE_IN_STEPS)
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
    assert!(
        indices.dims().is_scalar() || (indices.dims().len() == 2 && indices.dims()[0] == 1),
        "Unsupported indices shape: {:?}",
        indices.dims()
    );

    if indices.dims().is_scalar() {
        let axis = gather.axis as usize;
        assert_eq!(axis, 1);
        assert_eq!(data.dims().len(), 3);
        assert_eq!(data.dims()[0], 1);

        let gathered =
            &data.slice_at::<f32>(&[0, indices.data::<i64>()[0] as usize])[..data.dims()[2]];
        output.data_mut().copy_from_slice(gathered);
    } else {
        let axis = gather.axis as usize;
        assert_eq!(axis, 0);

        let indices = indices.data::<i64>();
        for (&i, o) in indices
            .iter()
            .zip(output.data_mut::<f32>().chunks_mut(data.dims()[1]))
        {
            assert!(i >= 0);
            o.copy_from_slice(&data.slice_at::<f32>(&[i as usize])[..data.dims()[1]]);
        }
    }
}

fn compute_reshape(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Op::RESHAPE_IN];
    let output = &mut outputs[Op::RESHAPE_OUT];
    output.copy_data_from(input)
}

fn compute_flatten(_flatten: &Flatten, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Op::FLATTEN_IN];
    let output = &mut outputs[Op::FLATTEN_OUT];
    output.copy_data_from(input);
}
