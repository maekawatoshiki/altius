mod conv2d;

use super::SessionError;
use altius_core::{
    model::Model,
    node::{
        compute_output_shapes, BatchNormalization, Cast, Concat, Flatten, Gemm, HardSigmoid,
        LeakyReLU, MaxPool, Node, NodeId, Op, ReduceMean, Resize, Squeeze, Transpose,
    },
    tensor::{Tensor, TensorElemType, TypedShape},
    value::ValueId,
};
use conv2d::Conv2dCtx;
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;
use ndarray::{linalg, s, Array2, ArrayView2, ArrayView3, ArrayView4};
use rustc_hash::FxHashMap;

use std::simd::Simd;
use std::time::{Duration, Instant};

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
    dummy_value: Tensor,
}

impl<'a> Interpreter<'a> {
    pub fn new(model: &'a Model) -> Self {
        let sorted_nodes = model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        infer_shapes(model, &sorted_nodes, &mut inferred_shapes);

        Interpreter {
            model,
            #[cfg(feature = "cuda")]
            cudnn_ctx: SafeCudnnContext(CudnnContext::new().expect("cudnn context init failed")),
            sorted_nodes,
            inferred_shapes,
            enable_profiling: false,
            dummy_value: Tensor::zeros::<f32>(vec![0].into()),
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
        if self.model.outputs.len() > 1 {
            log::debug!("Number of outputs: {}", self.model.outputs.len());
        }

        let mut profile = FxHashMap::default();
        let mut values = self.model.inits.clone();

        // Set inputs.
        for (id, tensor) in inputs {
            values.insert(id, tensor);
        }

        for &node in &self.sorted_nodes {
            self.run_node(&mut profile, &mut values, node);
        }

        if self.enable_profiling {
            log::info!(
                "Total execution time: {:#?}",
                profile.values().sum::<Duration>()
            );
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
            .map(|TypedShape { elem_ty, dims }| Tensor::zeros_of_type(elem_ty, dims))
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
            Op::Add => compute_add(node, &inputs, &mut outputs),
            Op::Sub => compute_sub(node, &inputs, &mut outputs),
            Op::Mul => compute_mul(node, &inputs, &mut outputs),
            Op::Div => compute_div(node, &inputs, &mut outputs),
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
            Op::Sigmoid => compute_sigmoid(&inputs, &mut outputs),
            Op::Erf => todo!("Erf"),
            Op::Clip => todo!("clip"),
            Op::Softmax(_) => todo!("Softmax"),
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
            Op::Gather(_) => todo!("Gather"),
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

fn compute_add(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input_a = inputs[Node::ADD_IN_A];
    let input_b = inputs[Node::ADD_IN_B];
    let output = &mut outputs[Node::ADD_OUT];

    if input_a.dims() == input_b.dims() {
        let mut input_a = input_a.data::<f32>();
        let mut input_b = input_b.data::<f32>();
        let mut output = output.data_mut::<f32>();
        let mut len = output.len();
        const SIMD_LEN: usize = 4;

        while len >= SIMD_LEN {
            let a = Simd::<f32, SIMD_LEN>::from_slice(&input_a[0..SIMD_LEN]);
            let b = Simd::<f32, SIMD_LEN>::from_slice(&input_b[0..SIMD_LEN]);
            let c = a + b;
            output[0..SIMD_LEN].copy_from_slice(c.as_array());
            input_a = &input_a[SIMD_LEN..];
            input_b = &input_b[SIMD_LEN..];
            output = &mut output[SIMD_LEN..];
            len -= SIMD_LEN
        }

        for (i, (a, b)) in input_a.iter().zip(input_b.iter()).enumerate() {
            output[i] = a + b;
        }

        return;
    }

    if input_a.dims().len() == 4 && input_b.dims().len() == 3 {
        assert!(input_a.dims()[1] == input_b.dims()[0]);
        assert!(input_b.dims()[1] == 1);
        assert!(input_b.dims()[2] == 1);

        for n in 0..input_a.dims()[0] {
            for z in 0..input_a.dims()[1] {
                for x in 0..input_a.dims()[2] {
                    for y in 0..input_a.dims()[3] {
                        *output.at_4d_mut(n, z, x, y) =
                            input_a.at_4d(n, z, x, y) + input_b.at_3d(z, 0, 0);
                    }
                }
            }
        }
    }
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
        let n = dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        for i in 0..max {
            output[i] = input_a[i] - input_b[i / n];
        }

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

    if input_a.dims() == input_b.dims() {
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

    let in_a = input_a.dims();
    let in_b = input_b.dims();
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

    if in_b.len() == 1 && in_a[in_a.len() - 1] == in_b[0] {
        let dims = input_a.dims();
        let len = dims.total_elems();
        let max = dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        for i in 0..len {
            output[i] = input_a[i] * input_b[i % max];
        }

        return;
    }

    todo!(
        "A shape: {:?}, B shape: {:?}",
        input_a.dims(),
        input_b.dims()
    )
}

fn compute_div(_node: &Node, inputs: &[&Tensor], outputs: &mut [Tensor]) {
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
        let dims = input_a.dims();
        let max = dims.total_elems();
        let n = dims.0.last().unwrap();
        let input_a = input_a.data::<f32>();
        let input_b = input_b.data::<f32>();
        let output = output.data_mut::<f32>();

        for i in 0..max {
            output[i] = input_a[i] / input_b[i / n];
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
    let output = &mut outputs[0];

    if input_a.dims() == input_b.dims() {
        for (i, (a, b)) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .enumerate()
        {
            output.data_mut::<f32>()[i] = a.powf(*b);
        }
        return;
    }

    // TODO: We need multidirectional broadcast!

    if input_b.dims().is_scalar() {
        let b = input_b.data::<f32>()[0];
        let output = output.data_mut::<f32>();

        for (a, o) in input_a.data::<f32>().iter().zip(output.iter_mut()) {
            *o = a.powf(b);
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
        // TODO: Don't use ndarray.
        let input_a = inputs[Node::GEMM_IN_A];
        let [_, m, _k] = input_a.fixed_dims::<3>();
        let [k, n] = input_b.fixed_dims::<2>();
        let output = &mut outputs[Node::GEMM_OUT];

        for i in 0..adim.len() {
            let a = ArrayView2::from_shape([m, k], input_a.slice_at(&[i])).unwrap();
            let b = ArrayView2::from_shape([k, n], input_b.data::<f32>()).unwrap();
            let mut c = Array2::zeros([m, n]);
            linalg::general_mat_mul(1., &a, &b, 0., &mut c);
            output.slice_at_mut(&[i])[..(m * n)].copy_from_slice(c.as_slice().unwrap());
        }
    } else if adim.len() == 3 && bdim.len() == 3 {
        // TODO: Don't use ndarray.
        let input_a = inputs[Node::GEMM_IN_A];
        let [_, m, _k] = input_a.fixed_dims::<3>();
        let [_, k, n] = input_b.fixed_dims::<3>();
        let output = &mut outputs[Node::GEMM_OUT];

        for i in 0..adim.len() {
            let a = ArrayView2::from_shape([m, k], input_a.slice_at(&[i])).unwrap();
            let b = ArrayView2::from_shape([k, n], input_b.slice_at(&[i])).unwrap();
            let mut c = Array2::zeros([m, n]);
            linalg::general_mat_mul(1., &a, &b, 0., &mut c);
            output.slice_at_mut(&[i])[..(m * n)].copy_from_slice(c.as_slice().unwrap());
        }
    } else {
        // TODO: Why don't use gemm library?
        for i in 0..input_a.dims()[0] {
            for j in 0..input_b.dims()[1] {
                let mut t = 0.0;
                for k in 0..input_b.dims()[0] {
                    t += input_a.at_2d(i, k) * input_b.at_2d(k, j);
                }
                *output.at_2d_mut(i, j) = t;
            }
        }
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

    let a =
        Array2::from_shape_vec(input_a.fixed_dims::<2>(), input_a.data::<f32>().to_vec()).unwrap();
    let b =
        Array2::from_shape_vec(input_b.fixed_dims::<2>(), input_b.data::<f32>().to_vec()).unwrap();
    let a = if gemm.trans_a { a.t() } else { a.view() };
    let b = if gemm.trans_b { b.t() } else { b.view() };

    let mut c = Array2::from_shape_vec([1, input_c.dims()[0]], input_c.data::<f32>().to_vec())
        .unwrap()
        .broadcast(output.fixed_dims::<2>())
        .unwrap()
        .into_owned();
    linalg::general_mat_mul(gemm.alpha, &a, &b, gemm.beta, &mut c);

    output.set_raw_vec(c.into_raw_vec());
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

fn compute_sigmoid(inputs: &[&Tensor], outputs: &mut [Tensor]) {
    let input: &[f32] = inputs[Node::SIGMOID_IN].data();
    let output: &mut [f32] = outputs[Node::SIGMOID_OUT].data_mut();

    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = 1. / (1. + (-i).exp())
    }
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
            in_view.permuted_axes([transpose.perm[0] as usize, transpose.perm[1] as usize]);
            output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
        }
        3 => {
            let in_view =
                ArrayView3::from_shape(input.fixed_dims::<3>(), input.data::<f32>()).unwrap();
            in_view.permuted_axes([
                transpose.perm[0] as usize,
                transpose.perm[1] as usize,
                transpose.perm[2] as usize,
            ]);
            output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
        }
        4 => {
            let in_view =
                ArrayView4::from_shape(input.fixed_dims::<4>(), input.data::<f32>()).unwrap();
            in_view.permuted_axes([
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
