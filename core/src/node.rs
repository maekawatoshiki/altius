use crate::{
    dim::Dimensions,
    tensor::{Tensor, TensorElemType},
    value::ValueId,
};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Conv2d(Conv2d),
    Add,
    Sub,
    Mul,
    Div,
    ReLU,
    LeakyReLU(LeakyReLU),
    Sigmoid,
    Cast(Cast),
    MaxPool(MaxPool),
    GlobalAveragePool,
    Reshape,
    Flatten(Flatten),
    Resize(Resize),
    Concat(Concat),
    Transpose(Transpose),
    Squeeze(Squeeze),
    Unsqueeze(Unsqueeze),
    ReduceMin(ReduceMin),
    Round,
    Exp,
    Loop,
    Tile,
    Slice,
    NonMaxSuppression,
    MatMul,
    Gemm(Gemm),
    HardSigmoid(HardSigmoid),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Conv2d {
    pub auto_pad: String,
    pub group: i64,
    pub kernel_shape: Dimensions,
    pub strides: Dimensions,
    pub padding: Dimensions,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaxPool {
    pub auto_pad: String,
    pub kernel_shape: Dimensions,
    pub strides: Dimensions,
    pub padding: Dimensions,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct HardSigmoid {
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LeakyReLU {
    pub alpha: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cast {
    pub to: TensorElemType,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Flatten {
    pub axis: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Resize {
    pub coordinate_transformation_mode: String,
    pub cubic_coeff_a: f32,
    pub exclude_outside: i64,
    pub extrapolation_value: f32,
    pub mode: String,
    pub nearest_mode: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Concat {
    pub axis: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transpose {
    pub perm: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Squeeze {
    pub axes: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Unsqueeze {
    pub axes: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReduceMin {
    pub axes: Vec<i64>,
    pub keep_dims: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Gemm {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl Node {
    pub const CONV2D_IN: usize = 0;
    pub const CONV2D_WEIGHT: usize = 1;
    pub const CONV2D_BIAS: usize = 2;
    pub const CONV2D_OUT: usize = 0;

    pub const ADD_IN_A: usize = 0;
    pub const ADD_IN_B: usize = 1;
    pub const ADD_OUT: usize = 0;

    pub const SUB_IN_A: usize = 0;
    pub const SUB_IN_B: usize = 1;
    pub const SUB_OUT: usize = 0;

    pub const MUL_IN_A: usize = 0;
    pub const MUL_IN_B: usize = 1;
    pub const MUL_OUT: usize = 0;

    pub const DIV_IN_A: usize = 0;
    pub const DIV_IN_B: usize = 1;
    pub const DIV_OUT: usize = 0;

    pub const RELU_IN: usize = 0;
    pub const RELU_OUT: usize = 0;

    pub const LEAKYRELU_IN: usize = 0;
    pub const LEAKYRELU_OUT: usize = 0;

    pub const SIGMOID_IN: usize = 0;
    pub const SIGMOID_OUT: usize = 0;

    pub const CAST_IN: usize = 0;
    pub const CAST_OUT: usize = 0;

    pub const ROUND_IN: usize = 0;
    pub const ROUND_OUT: usize = 0;

    pub const EXP_IN: usize = 0;
    pub const EXP_OUT: usize = 0;

    pub const MAXPOOL_IN: usize = 0;
    pub const MAXPOOL_OUT: usize = 0;

    pub const GLOBALAVERAGEPOOL_IN: usize = 0;
    pub const GLOBALAVERAGEPOOL_OUT: usize = 0;

    pub const RESHAPE_IN: usize = 0;
    pub const RESHAPE_SHAPE: usize = 1;
    pub const RESHAPE_OUT: usize = 0;

    pub const FLATTEN_IN: usize = 0;
    pub const FLATTEN_OUT: usize = 0;

    pub const RESIZE_IN_X: usize = 0;
    pub const RESIZE_IN_ROI: usize = 1;
    pub const RESIZE_IN_SCALES: usize = 2;
    pub const RESIZE_IN_SIZES: usize = 3;
    pub const RESIZE_OUT: usize = 0;

    pub const CONCAT_IN: usize = 0; // variadic
    pub const CONCAT_OUT: usize = 0;

    pub const TRANSPOSE_IN: usize = 0;
    pub const TRANSPOSE_OUT: usize = 0;

    pub const SQUEEZE_IN: usize = 0;
    pub const SQUEEZE_OUT: usize = 0;

    pub const UNSQUEEZE_IN: usize = 0;
    pub const UNSQUEEZE_OUT: usize = 0;

    pub const REDUCEMIN_IN: usize = 0;
    pub const REDUCEMIN_OUT: usize = 0;

    pub const TILE_IN: usize = 0;
    pub const TILE_REPEATS: usize = 1;
    pub const TILE_OUT: usize = 0;

    pub const SLICE_IN_DATA: usize = 0;
    pub const SLICE_IN_STARTS: usize = 1;
    pub const SLICE_IN_ENDS: usize = 2;
    pub const SLICE_IN_AXES: usize = 3;
    pub const SLICE_IN_STEPS: usize = 4;
    pub const SLICE_OUT: usize = 0;

    pub const NMS_IN_BOXES: usize = 0;
    pub const NMS_IN_SCORES: usize = 1;
    pub const NMS_IN_MAX_OUTPUT_BOXES_PER_CLASS: usize = 2;
    pub const NMS_IN_NMS_IOU_THRESHOLD: usize = 3;
    pub const NMS_IN_NMS_SCORE_THRESHOLD: usize = 4;
    pub const NMS_OUT: usize = 0;

    pub const MATMUL_IN_A: usize = 0;
    pub const MATMUL_IN_B: usize = 1;
    pub const MATMUL_OUT: usize = 0;

    pub const GEMM_IN_A: usize = 0;
    pub const GEMM_IN_B: usize = 1;
    pub const GEMM_IN_C: usize = 2;
    pub const GEMM_OUT: usize = 0;

    pub const HARDSIGMOID_IN: usize = 0;
    pub const HARDSIGMOID_OUT: usize = 0;

    pub fn new(op: Op) -> Self {
        Self {
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_in(mut self, id: ValueId) -> Self {
        self.inputs.push(id);
        self
    }

    pub fn with_ins(mut self, mut ids: Vec<ValueId>) -> Self {
        self.inputs.append(&mut ids);
        self
    }

    pub fn with_out(mut self, id: ValueId) -> Self {
        self.outputs.push(id);
        self
    }

    pub fn with_outs(mut self, mut ids: Vec<ValueId>) -> Self {
        self.outputs.append(&mut ids);
        self
    }

    pub fn alloc(self, arena: &mut NodeArena) -> NodeId {
        arena.alloc(self)
    }
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Conv2d(_) => "Conv2d",
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mul => "Mul",
            Op::Div => "Div",
            Op::ReLU => "ReLU",
            Op::LeakyReLU(_) => "LeakyReLU",
            Op::Sigmoid => "Sigmoid",
            Op::Cast(_) => "Cast",
            Op::MaxPool(_) => "MaxPool",
            Op::GlobalAveragePool => "GlobalAveragePool",
            Op::Reshape => "Reshape",
            Op::Flatten(_) => "Flatten",
            Op::Resize(_) => "Resize",
            Op::Concat(_) => "Concat",
            Op::Transpose(_) => "Transpose",
            Op::Squeeze(_) => "Squeeze",
            Op::Unsqueeze(_) => "Unsqueeze",
            Op::ReduceMin(_) => "ReduceMin",
            Op::Round => "Round",
            Op::Exp => "Exp",
            Op::Loop => "Loop",
            Op::Tile => "Tile",
            Op::Slice => "Slice",
            Op::NonMaxSuppression => "NonMaxSuppression",
            Op::MatMul => "MatMul",
            Op::Gemm(_) => "Gemm",
            Op::HardSigmoid(_) => "HardSigmoid",
        }
    }
}

/// Computes the output shape for `op`.
/// `op` could be overwritten. (e.g. paddings given auto_pad)
pub fn compute_output_shapes(op: &mut Op, inputs: &[Tensor]) -> Vec<Dimensions> {
    let mut shapes = vec![];
    match op {
        Op::Conv2d(conv) => {
            let auto_pad = &conv.auto_pad;
            let kernel = &conv.kernel_shape;
            let stride = &conv.strides;
            let padding = &conv.padding;
            let input = inputs[Node::CONV2D_IN].dims();
            let weight = inputs[Node::CONV2D_WEIGHT].dims();

            let pad_h;
            let pad_w;
            if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                let out0 = (input[2] as f32 / stride[0] as f32).ceil() as usize;
                let out1 = (input[3] as f32 / stride[1] as f32).ceil() as usize;
                let pad0 =
                    ((out0 - 1) * stride[0] + ((kernel[0] - 1) * 1 + 1)).saturating_sub(input[2]);
                let pad1 =
                    ((out1 - 1) * stride[1] + ((kernel[1] - 1) * 1 + 1)).saturating_sub(input[3]);
                assert!(auto_pad == "SAME_UPPER");
                let new_padding = vec![pad0 / 2, pad1 / 2, pad0 - pad0 / 2, pad1 - pad1 / 2];
                conv.padding = new_padding.into();
                pad_h = pad0;
                pad_w = pad1;
            } else if padding.len() == 2 {
                pad_h = padding[0] * 2;
                pad_w = padding[1] * 2;
                conv.padding = vec![padding[0], padding[1], padding[0], padding[1]].into();
            } else if padding.len() == 4 {
                pad_h = padding[0] + padding[2];
                pad_w = padding[1] + padding[3];
            } else {
                todo!()
            }

            let h_in = input[2];
            let w_in = input[3];
            let output_shape = vec![
                input[0],
                weight[0],
                (h_in + pad_h - 1 * (kernel[0] - 1) - 1) / stride[0] + 1,
                (w_in + pad_w - 1 * (kernel[1] - 1) - 1) / stride[1] + 1,
            ];
            shapes.push(output_shape.into());
        }
        Op::Add => {
            let in_a = inputs[Node::ADD_IN_A].dims();
            let in_b = inputs[Node::ADD_IN_B].dims();
            assert!(
                in_a == in_b || {
                    in_a.len() == 4
                        && in_b.len() == 3
                        && in_a[1] == in_b[0]
                        && in_b[1] == 1
                        && in_b[2] == 1
                }
            ); // TODO: Support broadcasting.
            shapes.push(in_a.clone());
        }
        Op::Sub => {
            let in_a = inputs[Node::SUB_IN_A].dims();
            let in_b = inputs[Node::SUB_IN_B].dims();
            assert!(in_a == in_b);
            shapes.push(in_a.clone());
        }
        Op::Mul => {
            let in_a = inputs[Node::MUL_IN_A].dims();
            let in_b = inputs[Node::MUL_IN_B].dims();
            assert!(
                in_a == in_b || {
                    in_a.len() == 4
                        && in_b.len() == 4
                        && in_a[0] == in_b[0]
                        && in_a[1] == in_b[1]
                        && in_a[2] == 1
                        && in_a[3] == 1
                }
            ); // TODO: Support broadcasting.
            shapes.push(in_b.clone());
        }
        Op::Div => {
            let in_a = inputs[Node::DIV_IN_A].dims();
            let in_b = inputs[Node::DIV_IN_B].dims();
            assert!(in_a == in_b);
            shapes.push(in_a.clone());
        }
        Op::MaxPool(maxpool) => {
            let auto_pad = &maxpool.auto_pad;
            let kernel = &maxpool.kernel_shape;
            let stride = &maxpool.strides;
            let input = &inputs[Node::MAXPOOL_IN].dims();
            let mut padding = &maxpool.padding;

            if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                let out0 = (input[2] as f32 / stride[0] as f32).ceil() as usize;
                let out1 = (input[3] as f32 / stride[1] as f32).ceil() as usize;
                let pad0 =
                    ((out0 - 1) * stride[0] + ((kernel[0] - 1) * 1 + 1)).saturating_sub(input[2]);
                let pad1 =
                    ((out1 - 1) * stride[1] + ((kernel[1] - 1) * 1 + 1)).saturating_sub(input[3]);
                assert!(auto_pad == "SAME_UPPER");
                let new_padding = vec![pad0 / 2, pad1 / 2, pad0 - pad0 / 2, pad1 - pad1 / 2];
                maxpool.padding = new_padding.into();
                padding = &maxpool.padding;
            }

            let h_in = input[2];
            let w_in = input[3];
            let output_shape = vec![
                input[0],
                input[1],
                (h_in + (padding[0] + padding[2]) - 1 * (kernel[0] - 1) - 1) / stride[0] + 1,
                (w_in + (padding[1] + padding[3]) - 1 * (kernel[1] - 1) - 1) / stride[1] + 1,
            ];
            shapes.push(output_shape.into());
        }
        Op::GlobalAveragePool => {
            let input = &inputs[Node::GLOBALAVERAGEPOOL_IN].dims();
            assert!(input.len() == 4);
            shapes.push(vec![input[0], input[1], 1, 1].into());
        }
        Op::Reshape => {
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
                        inputs[Node::RESHAPE_IN].dims().total_elems() / other_dims_sz as usize
                    } else {
                        x as usize
                    }
                })
                .collect::<Vec<_>>();
            shapes.push(shape.into());
        }
        Op::Flatten(flatten) => {
            let dims = inputs[Node::FLATTEN_IN].dims();
            assert!(flatten.axis >= 0);
            let x: Dimensions = dims[..flatten.axis as usize].to_vec().into();
            let y: Dimensions = dims[flatten.axis as usize..].to_vec().into();
            shapes.push(vec![x.total_elems(), y.total_elems()].into())
        }
        Op::Resize(resize) => {
            // TODO: Support other cases.
            assert!(resize.coordinate_transformation_mode == "asymmetric");
            assert!(resize.mode == "nearest");
            assert!(resize.nearest_mode == "floor");
            assert!(inputs.len() == 3);
            let x = &inputs[Node::RESIZE_IN_X];
            assert!(x.dims().len() == 4);
            let roi = &inputs[Node::RESIZE_IN_ROI].data::<f32>();
            let scales = &inputs[Node::RESIZE_IN_SCALES].data::<f32>();
            shapes.push(
                vec![
                    (x.dims()[0] as f32 * (roi[4] - roi[0]) * scales[0]).floor() as usize,
                    (x.dims()[1] as f32 * (roi[5] - roi[1]) * scales[1]).floor() as usize,
                    (x.dims()[2] as f32 * (roi[6] - roi[2]) * scales[2]).floor() as usize,
                    (x.dims()[3] as f32 * (roi[7] - roi[3]) * scales[3]).floor() as usize,
                ]
                .into(),
            )
        }
        Op::Concat(concat) => {
            let mut dims = inputs[Node::CONCAT_IN].dims().clone();
            let mut sum = 0;
            for i in inputs {
                sum += i.dims()[concat.axis as usize];
            }
            dims.as_mut_slice()[concat.axis as usize] = sum;
            shapes.push(dims);
        }
        Op::Transpose(trans) => {
            assert!(!trans.perm.is_empty());
            let in_dims = inputs[Node::TRANSPOSE_IN].dims().as_slice();
            let mut dims = vec![0usize; in_dims.len()];
            for i in 0..in_dims.len() {
                dims[i] = in_dims[trans.perm[i] as usize];
            }
            shapes.push(dims.into());
        }
        Op::Squeeze(squeeze) => {
            assert!(!squeeze.axes.is_empty());
            assert!(squeeze.axes.iter().all(|&x| x >= 0));
            let in_dims = inputs[Node::SQUEEZE_IN].dims().as_slice();
            let mut dims = vec![];
            for (i, &x) in in_dims.iter().enumerate() {
                if squeeze.axes.contains(&(i as i64)) {
                    continue;
                }
                dims.push(x);
            }
            shapes.push(dims.into());
        }
        Op::Unsqueeze(unsqueeze) => {
            assert!(!unsqueeze.axes.is_empty());
            assert!(unsqueeze.axes.iter().all(|&x| x >= 0));
            let in_dims = inputs[Node::UNSQUEEZE_IN].dims().as_slice().to_vec();
            let mut dims = in_dims.clone();
            for &x in unsqueeze.axes.iter() {
                dims.insert(x as usize, 1);
            }
            shapes.push(dims.into());
        }
        Op::ReduceMin(rmin) => {
            let in_dims = inputs[Node::REDUCEMIN_IN].dims();
            let mut dims = vec![];
            for (i, &x) in in_dims.as_slice().iter().enumerate() {
                if rmin.axes.contains(&(i as i64)) {
                    if rmin.keep_dims == 1 {
                        dims.push(1);
                    }
                    continue;
                }
                dims.push(x);
            }
            if dims.is_empty() {
                dims.push(1);
            }
            shapes.push(dims.into());
        }
        Op::Loop => {
            assert!(inputs.len() == 3);
            let m = inputs[0].data::<i64>();
            let cond = inputs[1].data::<u8>();
            assert!(cond[0] == 1);
            let v_initial = inputs[2].data::<i32>();
            assert!(v_initial[0] == 0);
            shapes.push(vec![1].into());
            shapes.push(vec![m[0] as usize].into());
        }
        Op::Tile => {
            let in_dims = inputs[Node::TILE_IN].dims();
            let reps = inputs[Node::TILE_REPEATS].data::<i64>();
            let mut dims = vec![];
            for (i, &x) in in_dims.as_slice().iter().enumerate() {
                dims.push(x * reps[i as usize] as usize);
            }
            shapes.push(dims.into());
        }
        Op::Slice => {
            let in_data_dims = inputs[Node::SLICE_IN_DATA].dims();
            let in_starts = inputs[Node::SLICE_IN_STARTS].data::<i64>();
            let in_ends = inputs[Node::SLICE_IN_ENDS].data::<i64>();
            let in_axes = inputs[Node::SLICE_IN_AXES].data::<i64>();
            let in_steps = inputs[Node::SLICE_IN_STEPS].data::<i64>();
            assert!(in_starts.iter().all(|&x| x >= 0));
            assert!(in_ends.iter().all(|&x| x >= 0));
            assert!(in_axes.iter().all(|&x| x >= 0));
            assert!(in_steps.iter().all(|&x| x >= 0));
            let mut dims = in_data_dims.clone();
            for (((start, end), axis), step) in in_starts
                .iter()
                .zip(in_ends.iter())
                .zip(in_axes.iter())
                .zip(in_steps.iter())
            {
                let start = *start as usize;
                let end = *end as usize;
                let axis = *axis as usize;
                let step = *step as usize;
                let out_dim = (end - start) / step;
                assert!(out_dim > 0);
                dims[axis] = out_dim;
            }
            shapes.push(dims.into());
        }
        Op::NonMaxSuppression => {
            todo!()
        }
        Op::MatMul => {
            let in_a = &inputs[Node::MATMUL_IN_A].dims();
            let in_b = &inputs[Node::MATMUL_IN_B].dims();
            assert_eq!(in_a[1], in_b[0]);
            shapes.push(vec![in_a[0], in_b[1]].into());
        }
        Op::Gemm(gemm) => {
            let in_a = &inputs[Node::GEMM_IN_A].dims();
            let (in_a0, in_a1) = if gemm.trans_a {
                (in_a[1], in_a[0])
            } else {
                (in_a[0], in_a[1])
            };
            let in_b = &inputs[Node::GEMM_IN_B].dims();
            let (in_b0, in_b1) = if gemm.trans_b {
                (in_b[1], in_b[0])
            } else {
                (in_b[0], in_b[1])
            };
            assert_eq!(in_a1, in_b0);
            shapes.push(vec![in_a0, in_b1].into());
        }
        Op::ReLU => {
            let input = inputs[Node::RELU_IN].dims();
            shapes.push(input.clone());
        }
        Op::LeakyReLU(_) => {
            let input = inputs[Node::LEAKYRELU_IN].dims();
            shapes.push(input.clone());
        }
        Op::Sigmoid => {
            let input = inputs[Node::SIGMOID_IN].dims();
            shapes.push(input.clone());
        }
        Op::Cast(_) => {
            let input = inputs[Node::CAST_IN].dims();
            shapes.push(input.clone());
        }
        Op::HardSigmoid(_) => {
            let input = inputs[Node::HARDSIGMOID_IN].dims();
            shapes.push(input.clone());
        }
        Op::Round => {
            let input = inputs[Node::ROUND_IN].dims();
            shapes.push(input.clone());
        }
        Op::Exp => {
            let input = inputs[Node::EXP_IN].dims();
            shapes.push(input.clone());
        }
    }
    shapes
}
