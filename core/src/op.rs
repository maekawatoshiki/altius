use crate::{
    dim::Dimensions,
    tensor::{Tensor, TensorElemType},
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
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Conv2d {
    pub auto_pad: String,
    pub dilations: Dimensions,
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

impl Op {
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
        }
    }
}
