use crate::{dim::Dimensions, tensor::Tensor, value::ValueId};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub attrs: Vec<Attr>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Conv2d,
    Add,
    ReLU,
    MaxPool,
    Reshape,
    MatMul,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Attr {
    Shape(Dimensions),
    String(String),
}

impl Node {
    pub const CONV2D_ATTR_AUTO_PAD: usize = 0;
    pub const CONV2D_ATTR_KERNEL: usize = 1;
    pub const CONV2D_ATTR_STRIDE: usize = 2;
    pub const CONV2D_ATTR_PADDING: usize = 3;
    pub const CONV2D_IN: usize = 0;
    pub const CONV2D_WEIGHT: usize = 1;
    pub const CONV2D_OUT: usize = 0;

    pub const ADD_IN_A: usize = 0;
    pub const ADD_IN_B: usize = 1;
    pub const ADD_OUT: usize = 0;

    pub const RELU_IN: usize = 0;
    pub const RELU_OUT: usize = 0;

    pub const MAXPOOL_ATTR_KERNEL: usize = 0;
    pub const MAXPOOL_ATTR_STRIDE: usize = 1;
    pub const MAXPOOL_IN: usize = 0;
    pub const MAXPOOL_OUT: usize = 0;

    pub const RESHAPE_IN: usize = 0;
    pub const RESHAPE_SHAPE: usize = 1;
    pub const RESHAPE_OUT: usize = 0;

    pub const MATMUL_IN_A: usize = 0;
    pub const MATMUL_IN_B: usize = 1;
    pub const MATMUL_OUT: usize = 0;

    pub fn new(op: Op) -> Self {
        Self {
            op,
            attrs: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_attr(mut self, attr: Attr) -> Self {
        self.attrs.push(attr);
        self
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
        let id = arena.alloc(self);
        id
    }

    /// Computes the output shape for a node given `inputs`.
    /// `attrs` could be overwritten. (e.g. paddings given auto_pad)
    pub fn compute_output_shapes(&self, inputs: &[Tensor], attrs: &mut [Attr]) -> Vec<Dimensions> {
        assert!(self.attrs == attrs);

        let mut shapes = vec![];
        match self.op {
            Op::Conv2d => {
                let auto_pad = self.attrs[Self::CONV2D_ATTR_AUTO_PAD].as_string().unwrap();
                let kernel = self.attrs[Self::CONV2D_ATTR_KERNEL].as_shape().unwrap();
                let stride = self.attrs[Self::CONV2D_ATTR_STRIDE].as_shape().unwrap();
                let padding = self.attrs[Self::CONV2D_ATTR_PADDING].as_shape().unwrap();
                let kernel = kernel.as_slice();
                let stride = stride.as_slice();
                let mut padding = padding.as_slice();
                let input = inputs[Self::CONV2D_IN].dims().as_slice();
                let weight = inputs[Self::CONV2D_WEIGHT].dims().as_slice();

                if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                    let out0 = (input[2] as f32 / stride[0] as f32).ceil() as usize;
                    let out1 = (input[3] as f32 / stride[1] as f32).ceil() as usize;
                    let pad0 = ((out0 - 1) * stride[0] + ((kernel[0] - 1) * 1 + 1))
                        .saturating_sub(input[2]);
                    let pad1 = ((out1 - 1) * stride[1] + ((kernel[1] - 1) * 1 + 1))
                        .saturating_sub(input[3]);
                    assert!(auto_pad == "SAME_UPPER");
                    let new_padding = vec![pad0 / 2, pad1 / 2, pad0 - pad0 / 2, pad1 - pad1 / 2];
                    attrs[Self::CONV2D_ATTR_PADDING] = Attr::Shape(new_padding.into());
                    padding = attrs[Self::CONV2D_ATTR_PADDING]
                        .as_shape()
                        .unwrap()
                        .as_slice();
                }

                let h_in = input[2];
                let w_in = input[3];
                let output_shape = vec![
                    input[0],
                    weight[0],
                    (h_in + 2 * padding[0] - 1 * (kernel[0] - 1) - 1) / stride[0] + 1,
                    (w_in + 2 * padding[1] - 1 * (kernel[1] - 1) - 1) / stride[1] + 1,
                ];
                shapes.push(output_shape.into());
            }
            Op::Add => {
                let in_a = inputs[Self::ADD_IN_A].dims();
                let in_b = inputs[Self::ADD_IN_B].dims();
                assert!(
                    in_a == in_b || {
                        in_a.len() == 4
                            && in_b.len() == 3
                            && in_a.as_slice()[1] == in_b.as_slice()[0]
                            && in_b.as_slice()[1] == 1
                            && in_b.as_slice()[2] == 1
                    }
                ); // TODO: Support broadcasting.
                shapes.push(in_a.clone());
            }
            Op::MaxPool => {
                let kernel = self.attrs[Self::MAXPOOL_ATTR_KERNEL].as_shape().unwrap();
                let stride = self.attrs[Self::MAXPOOL_ATTR_STRIDE].as_shape().unwrap();
                let input = &inputs[Self::MAXPOOL_IN].dims();

                let h_in = input.as_slice()[2];
                let w_in = input.as_slice()[3];
                let output_shape = vec![
                    input.as_slice()[0],
                    input.as_slice()[1],
                    (h_in + 2 * 0 - 1 * (kernel.as_slice()[0] - 1) - 1) / stride.as_slice()[0] + 1,
                    (w_in + 2 * 0 - 1 * (kernel.as_slice()[1] - 1) - 1) / stride.as_slice()[1] + 1,
                ];
                shapes.push(output_shape.into());
            }
            Op::Reshape => {
                let shape = inputs[Self::RESHAPE_SHAPE]
                    .data()
                    .as_i64()
                    .unwrap()
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>();
                shapes.push(shape.into());
            }
            Op::MatMul => {
                let in_a = &inputs[Self::MATMUL_IN_A].dims();
                let in_b = &inputs[Self::MATMUL_IN_B].dims();
                assert_eq!(in_a.as_slice()[1], in_b.as_slice()[0]);
                shapes.push(vec![in_a.as_slice()[0], in_b.as_slice()[1]].into());
            }
            Op::ReLU => {
                let input = inputs[Self::RELU_IN].dims();
                shapes.push(input.clone());
            }
        }
        shapes
    }
}

impl Attr {
    pub fn as_shape(&self) -> Option<&Dimensions> {
        match self {
            Attr::Shape(ref x) => Some(x),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&String> {
        match self {
            Attr::String(ref x) => Some(x),
            _ => None,
        }
    }
}

impl From<Vec<usize>> for Attr {
    fn from(v: Vec<usize>) -> Self {
        Attr::Shape(Dimensions(v))
    }
}

impl From<Dimensions> for Attr {
    fn from(v: Dimensions) -> Self {
        Attr::Shape(v)
    }
}

impl<'a> From<&'a str> for Attr {
    fn from(s: &'a str) -> Self {
        Attr::String(s.to_string())
    }
}
