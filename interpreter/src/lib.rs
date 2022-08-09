mod conv2d;

use altius_core::{
    model::Model,
    node::{
        compute_output_shapes, Cast, Concat, Flatten, Gemm, HardSigmoid, LeakyReLU, MaxPool, Node,
        NodeId, Op, Squeeze, Transpose,
    },
    tensor::Tensor,
    value::ValueId,
};
use conv2d::Conv2dCtx;
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;
use ndarray::{linalg, Array2, ArrayView2, ArrayView4};
use rustc_hash::FxHashMap;
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
    values: FxHashMap<ValueId, Tensor>,
    profile: FxHashMap<&'static str, Duration>,
    #[cfg(feature = "cuda")]
    cudnn_ctx: SafeCudnnContext,
    enable_profiling: bool,
}

impl<'a> Interpreter<'a> {
    pub fn new(model: &'a Model) -> Self {
        Interpreter {
            model,
            values: FxHashMap::default(),
            profile: FxHashMap::default(),
            #[cfg(feature = "cuda")]
            cudnn_ctx: SafeCudnnContext(CudnnContext::new().expect("cudnn context init failed")),
            enable_profiling: false,
        }
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn reset_profile(&mut self) {
        self.profile.clear();
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    pub fn run(&mut self, inputs: Vec<(ValueId, Tensor)>) -> Vec<&Tensor> {
        // assert!(self.model.inputs.len() == 1);
        // assert!(self.model.outputs.len() == 1);
        if self.model.outputs.len() > 1 {
            log::debug!("Number of outputs: {}", self.model.outputs.len());
        }

        // Set input & initializer values.
        for (id, tensor) in inputs {
            self.values.insert(id, tensor);
        }
        self.values.extend(self.model.inits.clone().into_iter());

        let nodes = self.model.topo_sort_nodes();

        // println!("sorted nodes: {:?}", nodes);

        for node in nodes {
            self.run_node(node);
        }

        if self.enable_profiling {
            log::info!(
                "Total execution time: {:#?}",
                self.profile.values().sum::<Duration>()
            );
            log::info!("Profile: {:#?}", self.profile);
        }

        self.model
            .outputs
            .iter()
            .map(|id| &self.values[id])
            .collect()
    }

    fn run_node(&mut self, node: NodeId) {
        let node = &self.model.nodes[node];
        let mut inputs = vec![];
        for input in node.inputs.iter() {
            inputs.push(self.values[input].clone());
        }
        let mut op = node.op.clone();
        let output_shapes = compute_output_shapes(&mut op, &inputs);
        let mut outputs = vec![];
        for output_shape in output_shapes {
            outputs.push(Tensor::zeros::<f32>(output_shape));
        }

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
            Op::Sub => todo!("sub"),
            Op::Mul => compute_mul(node, &inputs, &mut outputs),
            Op::Div => compute_div(node, &inputs, &mut outputs),
            Op::MaxPool(ref maxpool) => compute_max_pool(maxpool, &inputs, &mut outputs),
            Op::GlobalAveragePool => compute_gavg_pool(node, &inputs, &mut outputs),
            Op::Reshape => compute_reshape(node, &inputs, &mut outputs),
            Op::Flatten(ref flatten) => compute_flatten(flatten, &inputs, &mut outputs),
            Op::MatMul => compute_mat_mul(node, &inputs, &mut outputs),
            Op::Gemm(ref gemm) => compute_gemm(gemm, &inputs, &mut outputs),
            Op::ReLU => compute_relu(node, &inputs, &mut outputs),
            Op::HardSigmoid(ref hs) => compute_hard_sigomid(hs, &inputs, &mut outputs),
            Op::LeakyReLU(ref leaky) => compute_leaky_relu(leaky, &inputs, &mut outputs),
            Op::Sigmoid => todo!("sigmoid"),
            Op::Resize(_) => todo!("resize"),
            Op::Concat(ref concat) => compute_concat(concat, &inputs, &mut outputs),
            Op::Transpose(ref trans) => compute_transpose(trans, &inputs, &mut outputs),
            Op::Squeeze(ref squeeze) => compute_squeeze(squeeze, &inputs, &mut outputs),
            Op::Unsqueeze(_) => todo!("unsqueeze"),
            Op::ReduceMin(_) => todo!("reduce min"),
            Op::Round => todo!("round"),
            Op::Exp => todo!("exp"),
            Op::Loop => compute_loop(node, &inputs, &mut outputs),
            Op::Tile => compute_tile(node, &inputs, &mut outputs),
            Op::Cast(ref cast) => compute_cast(cast, &inputs, &mut outputs),
            Op::Slice => todo!("slice"),
            Op::NonMaxSuppression => todo!("nms"),
        }

        let elapsed = start.elapsed();
        if self.enable_profiling {
            *self.profile.entry(op.name()).or_insert(Duration::ZERO) += elapsed;
        }

        for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
            self.values.insert(val, output);
        }
    }
}

fn compute_gavg_pool(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::GLOBALAVERAGEPOOL_IN];
    let output = &mut outputs[Node::GLOBALAVERAGEPOOL_OUT];

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    let area = (input.dims()[2] * input.dims()[3]) as f32;

    for n in 0..input.dims()[0] {
        for c in 0..input.dims()[1] {
            let mut sum = 0f32;
            for h in 0..input.dims()[2] {
                for w in 0..input.dims()[3] {
                    sum += input.at_4d(n, c, h, w);
                }
            }
            *output.at_4d_mut(n, c, 0, 0) = sum / area;
        }
    }
}

fn compute_max_pool(maxpool: &MaxPool, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::MAXPOOL_IN];
    let output = &mut outputs[Node::MAXPOOL_OUT];

    let kernel = &maxpool.kernel_shape;
    let stride = &maxpool.strides;

    assert!(input.dims().len() == 4);
    assert!(output.dims().len() == 4);

    for n in 0..output.dims()[0] {
        for z in 0..input.dims()[1] {
            let mut x = 0isize; // TODO: pad
            for ax in 0..output.dims()[2] {
                let mut y = 0isize; // TODO: pad
                for ay in 0..output.dims()[3] {
                    let mut max = f32::MIN;
                    for fx in 0..kernel[0] as isize {
                        for fy in 0..kernel[1] as isize {
                            let ox = x + fx;
                            let oy = y + fy;

                            if ox < 0
                                || oy < 0
                                || ox >= input.dims()[2] as isize
                                || oy >= input.dims()[3] as isize
                            {
                                continue;
                            }

                            let val = input.at_4d(n, z, ox as usize, oy as usize);

                            if val >= max {
                                max = val;
                            }
                        }
                    }
                    *output.at_4d_mut(n, z, ax, ay) = if max == f32::MIN { 0.0 } else { max };
                    y += stride[1] as isize
                }
                x += stride[0] as isize
            }
        }
    }
}

fn compute_add(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input_a = &inputs[Node::ADD_IN_A];
    let input_b = &inputs[Node::ADD_IN_B];
    let output = &mut outputs[Node::ADD_OUT];

    if input_a.dims() == input_b.dims() {
        for (i, (a, b)) in input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .enumerate()
        {
            output.data_mut::<f32>()[i] = a + b;
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

        return;
    }
}

fn compute_mul(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input_a = &inputs[Node::MUL_IN_A];
    let input_b = &inputs[Node::MUL_IN_B];
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

    todo!()
}

fn compute_div(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input_a = &inputs[Node::MUL_IN_A];
    let input_b = &inputs[Node::MUL_IN_B];
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

    todo!()
}

fn compute_mat_mul(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input_a = &inputs[Node::MATMUL_IN_A];
    let input_b = &inputs[Node::MATMUL_IN_B];
    let output = &mut outputs[Node::MATMUL_OUT];

    assert!(input_a.dims().len() == 2);
    assert!(input_b.dims().len() == 2);
    assert!(input_a.dims()[1] == input_b.dims()[0]);

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

fn compute_gemm(gemm: &Gemm, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input_a = &inputs[Node::GEMM_IN_A];
    let input_b = &inputs[Node::GEMM_IN_B];
    let input_c = &inputs[Node::GEMM_IN_C];
    let output = &mut outputs[Node::GEMM_OUT];

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

fn compute_relu(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::RELU_IN];
    let output = &mut outputs[Node::RELU_OUT];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = i.max(0.0);
    }
}

fn compute_hard_sigomid(hs: &HardSigmoid, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::HARDSIGMOID_IN];
    let output = &mut outputs[Node::HARDSIGMOID_OUT];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = (hs.alpha * i + hs.beta).min(1.0).max(0.0);
    }
}

fn compute_leaky_relu(leaky: &LeakyReLU, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::HARDSIGMOID_IN];
    let output = &mut outputs[Node::HARDSIGMOID_OUT];

    for (i, o) in input
        .data::<f32>()
        .iter()
        .zip(output.data_mut::<f32>().iter_mut())
    {
        *o = if *i < 0.0 { leaky.alpha * i } else { *i };
    }
}

fn compute_concat(_concat: &Concat, _inputs: &[Tensor], _outputs: &mut [Tensor]) {
    todo!("concat")
}

fn compute_transpose(transpose: &Transpose, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::TRANSPOSE_IN];
    let output = &mut outputs[Node::TRANSPOSE_OUT];
    assert!(input.elem_ty().is_f32());

    if input.dims().len() == 2 {
        let in_view = ArrayView2::from_shape(input.fixed_dims::<2>(), input.data::<f32>()).unwrap();
        in_view.permuted_axes([transpose.perm[0] as usize, transpose.perm[1] as usize]);
        output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
    } else if input.dims().len() == 4 {
        let in_view = ArrayView4::from_shape(input.fixed_dims::<4>(), input.data::<f32>()).unwrap();
        in_view.permuted_axes([
            transpose.perm[0] as usize,
            transpose.perm[1] as usize,
            transpose.perm[2] as usize,
            transpose.perm[3] as usize,
        ]);
        output.set_raw_vec(in_view.as_standard_layout().to_owned().into_raw_vec());
    } else {
        todo!()
    }
}

fn compute_squeeze(_squeeze: &Squeeze, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::SQUEEZE_IN];
    assert!(input.elem_ty().is_f32());
    let output = &mut outputs[Node::SQUEEZE_OUT];
    output.set_raw_vec(input.data::<f32>().to_vec());
}

fn compute_loop(_node: &Node, _inputs: &[Tensor], _outputs: &mut [Tensor]) {
    todo!("loop")
}

fn compute_tile(_node: &Node, _inputs: &[Tensor], _outputs: &mut [Tensor]) {
    todo!("tile")
}

fn compute_cast(cast: &Cast, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::CAST_IN];
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

fn compute_reshape(_node: &Node, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::RESHAPE_IN];
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

fn compute_flatten(_flatten: &Flatten, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::FLATTEN_IN];
    let output = &mut outputs[Node::FLATTEN_OUT];
    *output = input.clone().reshape_into(output.dims().clone());
}
