use super::{
    conv2d::{self, Conv2dCtx},
    exp::fast_exp,
    gemm::{sgemm, sgemm2},
    thread::ThreadCtx,
};

use crate::SessionError;
use altius_core::{
    model::Model,
    node::{
        compute_output_shapes, BatchNormalization, Cast, Concat, Flatten, Gather, Gemm,
        HardSigmoid, LeakyReLU, MaxPool, Node, NodeId, Op, ReduceMean, Resize, Softmax, Split,
        Squeeze, Transpose, Unsqueeze,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
    value::ValueId,
};
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;
use ndarray::{s, ArrayView, ArrayView2, ArrayView3, ArrayView4, ArrayView5, Axis, Dim, Ix};
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
use cuda::*;

pub struct InterpreterSession<'a> {
    pub(super) model: &'a Model,
    #[cfg(feature = "cuda")]
    pub(super) cudnn_ctx: SafeCudnnContext,
    pub(super) execution_plans: Vec<NodeExecutionPlan>,
    pub(super) inferred_shapes: FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    pub(super) enable_profiling: bool,
    pub(super) values: ThreadLocal<RefCell<FxHashMap<ValueId, Tensor>>>,
    pub(super) dummy_value: Tensor,
    pub(super) tctx: ThreadCtx,
}

/// Represents a node to execute and values to be freed after the execution of the node.
#[derive(Debug)]
pub(super) struct NodeExecutionPlan {
    /// The node to execute.
    node_id: NodeId,

    /// Values to be freed after the execution of the node.
    free_vals: Vec<ValueId>,
}

impl<'a> InterpreterSession<'a> {
    pub fn model(&self) -> &Model {
        self.model
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
            self.run_node(&mut profile, values, node.node_id);

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
                    let output_shapes =
                        compute_output_shapes(&mut op, &inputs, self.model.opset_version);
                    (op, output_shapes)
                });
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
            Op::Tanh => compute_tanh(&inputs, &mut outputs),
            Op::Clip => todo!("clip"),
            Op::Where => compute_where(&inputs, &mut outputs),
            Op::Softmax(ref softmax) => compute_softmax(&self.tctx, softmax, &inputs, &mut outputs),
            Op::Resize(ref resize) => compute_resize(&self.tctx, resize, &inputs, &mut outputs),
            Op::Concat(ref concat) => compute_concat(concat, &inputs, &mut outputs),
            Op::Transpose(ref trans) => compute_transpose(trans, &inputs, &mut outputs),
            Op::Squeeze(ref squeeze) => compute_squeeze(squeeze, &inputs, &mut outputs),
            Op::Unsqueeze(ref unsqueeze) => compute_unsqueeze(unsqueeze, &inputs, &mut outputs),
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
            Op::Split(ref split) => {
                compute_split(self.model.opset_version, split, &inputs, &mut outputs)
            }
            Op::Slice => compute_slice(node, &inputs, &mut outputs),
            Op::Gather(ref gather) => compute_gather(gather, &inputs, &mut outputs),
            Op::Shape(_) => todo!("shape"),
            Op::NonMaxSuppression => todo!("nms"),
            Op::Constant(_) => todo!("constant"),
        }

        #[cfg(not(target_arch = "wasm32"))]
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
            for ax in 0..output_w {
                let mut max = f32::MIN;
                for fy in 0..kernel_h {
                    let oy = y + fy as isize;
                    for fx in 0..kernel_w {
                        let ox = x + fx as isize;
                        if ox < 0 || oy < 0 || ox >= input_w as isize || oy >= input_h as isize {
                            continue;
                        }
                        max = input[oy as usize * input_w + ox as usize].max(max);
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

    const SIMD_LEN: usize = 8;

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
                        let mut input_a = input_a;
                        let mut input_b = input_b;
                        let mut output = output;
                        let mut len = output.len();

                        while len >= SIMD_LEN {
                            let a = Simd::<_, SIMD_LEN>::from_slice(input_a);
                            let b = Simd::<_, SIMD_LEN>::from_slice(input_b);
                            output[0..SIMD_LEN].copy_from_slice((a + b).as_ref());
                            (input_a, input_b, output) = (
                                &input_a[SIMD_LEN..],
                                &input_b[SIMD_LEN..],
                                &mut output[SIMD_LEN..],
                            );
                            len -= SIMD_LEN;
                        }

                        for ((a, b), o) in input_a.iter().zip(input_b.iter()).zip(output.iter_mut())
                        {
                            *o = a + b;
                        }
                    });
                });
        });

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
        } else if bdims.as_slice()[0..bdims.len() - 1].iter().all(|&d| d == 1)
            && adims.as_slice().last().unwrap() == bdims.as_slice().last().unwrap()
        {
            // e.g.
            //   a: [1, 12, 9, 9]
            //   b: [1,  1, 1, 9]
            (adims, bdims, input_a, input_b)
        } else {
            compute_general_add(input_a, input_b, output);
            return;
        };

    let input_a = input_a.data::<f32>();
    let input_b = input_b.data::<f32>();
    let output = output.data_mut::<f32>();
    let blen = *bdims.as_slice().last().unwrap();
    let batch = (100000 / blen).max(1);

    tctx.scope(|scope| {
        input_a
            .chunks(blen * batch)
            .zip(output.chunks_mut(blen * batch))
            .for_each(|(input_a, output)| {
                scope.spawn(move || {
                    input_a.chunks(blen).zip(output.chunks_mut(blen)).for_each(
                        |(input_a, output)| {
                            let mut input_a = input_a;
                            let mut input_b = input_b;
                            let mut output = output;
                            let mut len = output.len();

                            while len >= SIMD_LEN {
                                let a = Simd::<_, SIMD_LEN>::from_slice(&input_a);
                                let b = Simd::<_, SIMD_LEN>::from_slice(&input_b);
                                let c = a + b;
                                output[0..SIMD_LEN].copy_from_slice(c.as_ref());
                                input_a = &input_a[SIMD_LEN..];
                                input_b = &input_b[SIMD_LEN..];
                                output = &mut output[SIMD_LEN..];
                                len -= SIMD_LEN
                            }

                            for ((a, b), o) in
                                input_a.iter().zip(input_b.iter()).zip(output.iter_mut())
                            {
                                *o = a + b;
                            }
                        },
                    );
                })
            });
    });
}

fn compute_general_add(input_a: &Tensor, input_b: &Tensor, output: &mut Tensor) {
    if output.dims().len() == 4 {
        let [odim0, odim1, odim2, odim3] = output.fixed_dims::<4>();
        let out_shape = output.dims().as_slice().to_vec();
        let [astr0, astr1, astr2, astr3] = input_a
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<4>();
        let [bstr0, bstr1, bstr2, bstr3] = input_b
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<4>();

        let mut input_a0 = input_a.data::<f32>().as_ptr();
        let mut input_b0 = input_b.data::<f32>().as_ptr();
        let mut output = output.data_mut::<f32>().as_mut_ptr();

        for _ in 0..odim0 {
            let (mut input_a1, mut input_b1) = (input_a0, input_b0);
            for _ in 0..odim1 {
                let (mut input_a2, mut input_b2) = (input_a1, input_b1);
                for _ in 0..odim2 {
                    let (mut input_a3, mut input_b3) = (input_a2, input_b2);
                    for _ in 0..odim3 {
                        unsafe { *output = *input_a3 + *input_b3 };
                        (output, input_a3, input_b3) =
                            unsafe { (output.add(1), input_a3.add(astr3), input_b3.add(bstr3)) };
                    }
                    (input_a2, input_b2) = unsafe { (input_a2.add(astr2), input_b2.add(bstr2)) };
                }
                (input_a1, input_b1) = unsafe { (input_a1.add(astr1), input_b1.add(bstr1)) };
            }
            (input_a0, input_b0) = unsafe { (input_a0.add(astr0), input_b0.add(bstr0)) };
        }
        return;
    }

    if output.dims().len() == 3 {
        let [odim0, odim1, odim2] = output.fixed_dims::<3>();
        let out_shape = output.dims().as_slice().to_vec();
        let [astr0, astr1, astr2] = input_a
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<3>();
        let [bstr0, bstr1, bstr2] = input_b
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<3>();

        let mut input_a0 = input_a.data::<f32>().as_ptr();
        let mut input_b0 = input_b.data::<f32>().as_ptr();
        let mut output = output.data_mut::<f32>().as_mut_ptr();

        for _ in 0..odim0 {
            let (mut input_a1, mut input_b1) = (input_a0, input_b0);
            for _ in 0..odim1 {
                let (mut input_a2, mut input_b2) = (input_a1, input_b1);
                for _ in 0..odim2 {
                    unsafe { *output = *input_a2 + *input_b2 };
                    (output, input_a2, input_b2) =
                        unsafe { (output.add(1), input_a2.add(astr2), input_b2.add(bstr2)) };
                }
                (input_a1, input_b1) = unsafe { (input_a1.add(astr1), input_b1.add(bstr1)) };
            }
            (input_a0, input_b0) = unsafe { (input_a0.add(astr0), input_b0.add(bstr0)) };
        }
        return;
    }

    if output.dims().len() == 2 {
        let [odim0, odim1] = output.fixed_dims::<2>();
        let out_shape = output.dims().as_slice().to_vec();
        let [astr0, astr1] = input_a
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<2>();
        let [bstr0, bstr1] = input_b
            .strides_for_broadcasting(&out_shape)
            .unwrap()
            .to_fixed_dims::<2>();

        let mut input_a0 = input_a.data::<f32>().as_ptr();
        let mut input_b0 = input_b.data::<f32>().as_ptr();
        let mut output = output.data_mut::<f32>().as_mut_ptr();

        for _ in 0..odim0 {
            let (mut input_a1, mut input_b1) = (input_a0, input_b0);
            for _ in 0..odim1 {
                unsafe { *output = *input_a1 + *input_b1 };
                (output, input_a1, input_b1) =
                    unsafe { (output.add(1), input_a1.add(astr1), input_b1.add(bstr1)) };
            }
            (input_a0, input_b0) = unsafe { (input_a0.add(astr0), input_b0.add(bstr0)) };
        }
        return;
    }
}

fn compute_sub(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::SUB_IN_A];
    let input_b = inputs[Node::SUB_IN_B];
    let output = &mut outputs[Node::SUB_OUT];
    const SIMD_LEN: usize = 8;

    if input_a.dims() == input_b.dims() {
        let mut input_a = input_a.data::<f32>();
        let mut input_b = input_b.data::<f32>();
        let mut output = output.data_mut::<f32>();
        let mut len = output.len();

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

    if input_a.dims().is_scalar() {
        let input_a = input_a.data::<f32>()[0];
        let mut input_b = input_b.data::<f32>();
        let mut output = output.data_mut::<f32>();

        let mut len = output.len();

        while len >= SIMD_LEN {
            let a = Simd::splat(input_a);
            let b = Simd::<_, SIMD_LEN>::from_slice(input_b);
            output[0..SIMD_LEN].copy_from_slice((a - b).as_ref());
            (input_b, output) = (&input_b[SIMD_LEN..], &mut output[SIMD_LEN..]);
            len -= SIMD_LEN;
        }

        for (b, o) in input_b.iter().zip(output.iter_mut()) {
            *o = input_a - b;
        }

        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_a.dims().len() == 3
        && input_b.dims().len() == 3
        && input_b.dims()[input_b.dims().len() - 1] == 1
    {
        let dims = input_a.dims();
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

    const SIMD_LEN: usize = 8;

    if input_a.dims() == input_b.dims() {
        let mut input_a = input_a.data::<f32>();
        let mut input_b = input_b.data::<f32>();
        let mut output = output.data_mut::<f32>();

        let mut len = input_a.len();

        while len >= SIMD_LEN {
            let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
            let b = Simd::<f32, SIMD_LEN>::from_slice(&input_b[0..SIMD_LEN]).recip();
            output[0..SIMD_LEN].copy_from_slice((a * b).as_ref());
            input_a = &input_a[SIMD_LEN..];
            input_b = &input_b[SIMD_LEN..];
            output = &mut output[SIMD_LEN..];
            len -= SIMD_LEN
        }

        for ((a, b), o) in input_a.iter().zip(input_b.iter()).zip(output.iter_mut()) {
            *o = a / b;
        }
        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_a.dims().len() == 3
        && input_b.dims().len() == 3
        && input_b.dims()[input_b.dims().len() - 1] == 1
    {
        let dims = input_a.dims();
        let n = *dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        if n < 100000 {
            input_a
                .chunks(n)
                .zip(input_b.iter())
                .zip(output.chunks_mut(n))
                .for_each(|((mut a, &b), mut o)| {
                    let mut len = a.len();
                    let b_ = Simd::<f32, SIMD_LEN>::splat(b).recip();

                    while len >= SIMD_LEN {
                        let a_ = Simd::<f32, SIMD_LEN>::from_slice(&a[0..SIMD_LEN]);
                        o[0..SIMD_LEN].copy_from_slice((a_ * b_).as_ref());
                        a = &a[SIMD_LEN..];
                        o = &mut o[SIMD_LEN..];
                        len -= SIMD_LEN
                    }

                    for (a, o) in a.iter().zip(o.iter_mut()) {
                        *o = a * b.recip();
                    }
                });
        } else {
            tctx.scope(|scope| {
                input_a
                    .chunks(n)
                    .zip(input_b.iter())
                    .zip(output.chunks_mut(n))
                    .for_each(|((a, &b), o)| {
                        scope.spawn(move || {
                            for (&a, o) in a.iter().zip(o.iter_mut()) {
                                *o = a * b.recip();
                            }
                        });
                    });
            });
        }

        return;
    }

    if input_b.dims().is_scalar() {
        let mut input_a = input_a.data::<f32>();
        let b = input_b.data::<f32>()[0].recip();
        let simd_b = Simd::<f32, SIMD_LEN>::splat(b);
        let mut output = output.data_mut::<f32>();
        let mut len = input_a.len();

        while len >= SIMD_LEN {
            let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
            output[0..SIMD_LEN].copy_from_slice((a * simd_b).as_ref());
            input_a = &input_a[SIMD_LEN..];
            output = &mut output[SIMD_LEN..];
            len -= SIMD_LEN
        }

        for (a, o) in input_a.iter().zip(output.iter_mut()) {
            *o = a * b;
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
    let input_a = inputs[Node::MATMUL_IN_A];
    let input_b = inputs[Node::MATMUL_IN_B];
    let output = &mut outputs[Node::MATMUL_OUT];

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
    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = if *i < 0.0 { leaky.alpha * i } else { *i };
    }
}

fn compute_gelu(tctx: &ThreadCtx, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    const B: f32 = 0.7978845608028654f32; // sqrt(2.0 / PI)
    const C: f32 = 0.035677408136300125f32; // 0.044715 * sqrt(2.0 / PI)

    let input: &[f32] = inputs[0].data();
    let output: &mut [f32] = outputs[0].data_mut();
    let n = tctx.num_threads();

    tctx.scope(|scope| {
        input
            .chunks(input.len() / n)
            .zip(output.chunks_mut(input.len() / n))
            .for_each(|(input, output)| {
                scope.spawn(move || {
                    for (&i, o) in input.iter().zip(output.iter_mut()) {
                        *o = i * (C * i * i + B);
                    }
                    tanh(output);
                    for (&i, o) in input.iter().zip(output.iter_mut()) {
                        *o = (*o + 1.) * (i * 0.5);
                    }
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

    let axis_len = *input.dims().as_slice().last().unwrap();
    let input = input.data::<f32>();
    let output = output.data_mut::<f32>();

    let n = tctx.num_threads();
    let chunk = if input.len() < n {
        input.len()
    } else {
        input.len() / n
    };

    tctx.scope(|scope| {
        input
            .chunks(chunk)
            .zip(output.chunks_mut(chunk))
            .for_each(|(input, output)| scope.spawn(|| fast_exp(output, input)));
    });

    let batch = (output.len() / 100000).max(1); // 100000 is magic number :(
                                                // I think processing more than 100000 elements for
                                                // each core is just right.

    tctx.scope(|scope| {
        output.chunks_mut(axis_len * batch).for_each(|output| {
            scope.spawn(|| {
                output.chunks_mut(axis_len).for_each(|output| {
                    let sum = fast_sum(output);
                    output.iter_mut().for_each(|o| *o /= sum);
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
        sum += Simd::<f32, SIMD_LEN>::from_slice(&slice);
        slice = &slice[SIMD_LEN..];
        len -= SIMD_LEN
    }

    sum.reduce_sum() + slice.iter().sum::<f32>()
}

fn compute_resize(tctx: &ThreadCtx, resize: &Resize, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    assert!(matches!(inputs.len(), 3 | 4));

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

    if resize.mode == "nearest" {
        let mut input = input.data::<f32>();
        let mut output = output.data_mut::<f32>();

        for _ in 0..outer {
            for (h, o) in (0..output_h).zip(output.chunks_mut(output_w)) {
                let ih = (h as f32 / scale) as usize;
                let ihw = input_w * ih;
                for (w, o) in (0..output_w).zip(o.iter_mut()) {
                    let iwf = (w as f32 / scale) as usize;
                    let iw = iwf as usize;
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

fn compute_concat(concat: &Concat, inputs: &[&Tensor], outputs: &mut [Tensor]) {
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

    let output = &mut outputs[Node::CONCAT_OUT];
    let d = output.dims().len();

    output.set_raw_vec(match d {
        1 => concat!(1),
        2 => concat!(2),
        3 => concat!(3),
        4 => concat!(4),
        5 => concat!(5),
        _ => todo!("Concat: Unsupported shape."),
    });
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
    let output = &mut outputs[Node::SQUEEZE_OUT];
    output.copy_data_from(&input);
}

fn compute_unsqueeze(_: &Unsqueeze, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[0];
    let output = &mut outputs[0];
    output.copy_data_from(&input);
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
    } else if input.elem_ty().is_i64() && cast.to.is_f32() {
        let input = input.data::<i64>();
        let output = output.data_mut::<f32>();
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = *i as f32;
        }
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
            output[offset[i]..offset[i] + input.len()].copy_from_slice(&input);
            offset[i] += *sp as usize;
            s += *sp as usize
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
    let input = inputs[Node::RESHAPE_IN];
    let output = &mut outputs[Node::RESHAPE_OUT];
    output.copy_data_from(input)
}

fn compute_flatten(_flatten: &Flatten, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input = inputs[Node::FLATTEN_IN];
    let output = &mut outputs[Node::FLATTEN_OUT];
    output.copy_data_from(input);
}

// TODO: Better move to another file.
/// Infer `TypedShape`s of output tensors for each node.
/// It skips to infer on nodes without information for inference.
pub(super) fn infer_shapes(
    model: &Model,
    sorted_nodes: &[NodeId],
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
) {
    let mut values = model.inits.clone();

    for &val_id in &model.inputs {
        let shape = &model.values.inner()[val_id].shape;
        let Some(shape) = shape else { continue };
        let tensor = Tensor::zeros_of_type(shape.elem_ty, shape.dims.clone());
        values.insert(val_id, tensor);
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
        let Some(input) = values.get(input) else { return };
        inputs.push(input);
    }
    let output_shapes = compute_output_shapes(&mut op, &inputs, model.opset_version);
    let mut outputs = vec![];
    for shape in &output_shapes {
        outputs.push(Tensor::empty_of_type(shape.elem_ty, shape.dims.clone()));
    }
    for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
        values.insert(val, output);
    }
    shapes.insert(node_id, (op, output_shapes));
}

pub(super) fn create_execution_plan(
    model: &Model,
    sorted_nodes: &[NodeId],
) -> Vec<NodeExecutionPlan> {
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
                .or_insert_with(|| vec![])
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
