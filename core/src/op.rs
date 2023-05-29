use crate::{
    fixed_dim::FixedDimensions,
    tensor::{Tensor, TensorElemType},
    value::ValueId,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Conv2d(Conv2d),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Greater,
    Sqrt,
    ReLU,
    Gelu,
    LeakyReLU(LeakyReLU),
    Sigmoid,
    Erf,
    Tanh,
    Clip,
    Where,
    Softmax(Softmax),
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
    ReduceMax(ReduceMax),
    ReduceMean(ReduceMean),
    Round,
    Exp,
    Expand,
    Range,
    Loop,
    Tile,
    Split(Split),
    Slice,
    Gather(Gather),
    Shape(Shape),
    NonMaxSuppression,
    MatMul,
    Gemm(Gemm),
    BatchNormalization(BatchNormalization),
    LayerNormalization(LayerNormalization),
    HardSigmoid(HardSigmoid),
    Constant(Constant),
    FusedElemwise(FusedElemwise), // This is not part of the ONNX spec.
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Conv2d {
    pub auto_pad: String,
    pub dilations: FixedDimensions,
    pub group: i64,
    pub kernel_shape: FixedDimensions,
    pub strides: FixedDimensions,
    pub padding: FixedDimensions,
    pub activation: Option<FusedActivation>, // This is not part of the ONNX spec.
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusedActivation {
    Relu,
    HardSigmoid(HardSigmoid),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaxPool {
    pub auto_pad: String,
    pub kernel_shape: FixedDimensions,
    pub strides: FixedDimensions,
    pub padding: FixedDimensions,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HardSigmoid {
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LeakyReLU {
    pub alpha: f32,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax>
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Softmax {
    pub axis: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cast {
    pub to: TensorElemType,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Concat {
    pub axis: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transpose {
    pub perm: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Squeeze {
    pub axes: Vec<i64>,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Unsqueeze {
    pub axes: Vec<i64>, // From opset version 13, this attribute is no longer used.
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReduceMin {
    pub axes: Vec<i64>,
    pub keep_dims: bool,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReduceMax {
    pub axes: Vec<i64>,
    pub keep_dims: bool,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReduceMean {
    pub axes: Vec<i64>,
    pub keep_dims: bool,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Split {
    pub axis: i64,
    pub split: Vec<i64>, // From opset verion 13, this attribute is no longer used.
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Gather {
    pub axis: i64,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape>
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape {
    pub end: Option<i64>,
    pub start: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Gemm {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchNormalization {
    pub epsilon: f32,
    pub momentum: f32,
    pub training_mode: bool,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization>
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayerNormalization {
    pub axis: i64,
    pub epsilon: f32,
    pub stash_type: i64,
}

/// <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    pub value: Tensor,
    // TODO: Other attributes
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedElemwise {
    pub input_map: Vec<ValueId>,
    pub chain: Vec<(Op, Vec<ValueId>, Vec<ValueId>)>, // op, inputs, outputs
}

impl Op {
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

    pub const CLIP_IN: usize = 0;
    pub const CLIP_OUT: usize = 0;

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

    pub const BATCHNORM_IN_X: usize = 0;
    pub const BATCHNORM_IN_SCALE: usize = 1;
    pub const BATCHNORM_IN_B: usize = 2;
    pub const BATCHNORM_IN_INPUT_MEAN: usize = 3;
    pub const BATCHNORM_IN_INPUT_VAR: usize = 4;
    pub const BATCHNORM_OUT_Y: usize = 0;

    pub const HARDSIGMOID_IN: usize = 0;
    pub const HARDSIGMOID_OUT: usize = 0;

    pub fn name(&self) -> &'static str {
        match self {
            Op::Conv2d(_) => "Conv2d",
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mul => "Mul",
            Op::Div => "Div",
            Op::Pow => "Pow",
            Op::Greater => "Greater",
            Op::Sqrt => "Sqrt",
            Op::ReLU => "ReLU",
            Op::LeakyReLU(_) => "LeakyReLU",
            Op::Gelu => "Gelu",
            Op::Sigmoid => "Sigmoid",
            Op::Erf => "Erf",
            Op::Tanh => "Tanh",
            Op::Clip => "Clip",
            Op::Where => "Where",
            Op::Softmax(_) => "Softmax",
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
            Op::ReduceMax(_) => "ReduceMax",
            Op::ReduceMean(_) => "ReduceMean",
            Op::Round => "Round",
            Op::Exp => "Exp",
            Op::Expand => "Expand",
            Op::Range => "Range",
            Op::Loop => "Loop",
            Op::Tile => "Tile",
            Op::Split(_) => "Split",
            Op::Slice => "Slice",
            Op::Gather(_) => "Gather",
            Op::Shape(_) => "Shape",
            Op::NonMaxSuppression => "NonMaxSuppression",
            Op::MatMul => "MatMul",
            Op::Gemm(_) => "Gemm",
            Op::BatchNormalization(_) => "BatchNormalization",
            Op::LayerNormalization(_) => "LayerNormalization",
            Op::HardSigmoid(_) => "HardSigmoid",
            Op::Constant(_) => "Constant",
            Op::FusedElemwise(_) => "FusedElemwise",
        }
    }

    pub fn is_elemwise(&self) -> bool {
        // TODO: Support in the future
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Pow
                | Self::Greater
                | Self::Sqrt
                | Self::ReLU
                | Self::LeakyReLU(_)
                // | Self::Gelu
                // | Self::Sigmoid
                // | Self::Erf
                // | Self::Tanh
                // | Self::Clip
                // | Self::Where
                // | Self::Softmax(_)
                // | Self::Cast(_)
                // | Self::Round
                // | Self::Exp
                | Self::HardSigmoid(_)
        )
    }
}
