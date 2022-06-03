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

#[derive(Debug, Clone)]
pub enum Attr {
    Shape(Dimensions),
}

impl Node {
    pub const CONV2D_ATTR_KERNEL: usize = 0;
    pub const CONV2D_ATTR_STRIDE: usize = 1;
    pub const CONV2D_ATTR_PADDING: usize = 2;
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

    pub fn with_out(mut self, id: ValueId) -> Self {
        self.outputs.push(id);
        self
    }

    pub fn alloc(self, arena: &mut NodeArena) -> NodeId {
        let id = arena.alloc(self);
        id
    }

    pub fn compute_output_shapes(&self, inputs: &[Tensor]) -> Vec<Dimensions> {
        let mut shapes = vec![];
        match self.op {
            Op::Conv2d => {
                let Attr::Shape(kernel) = &self.attrs[Self::CONV2D_ATTR_KERNEL];
                let Attr::Shape(stride) = &self.attrs[Self::CONV2D_ATTR_STRIDE];
                let Attr::Shape(padding) = &self.attrs[Self::CONV2D_ATTR_PADDING];
                let input = inputs[Self::CONV2D_IN].dims();
                let weight = inputs[Self::CONV2D_WEIGHT].dims();

                let h_in = input.as_slice()[2];
                let w_in = input.as_slice()[3];
                let output_shape = vec![
                    input.as_slice()[0],
                    weight.as_slice()[0],
                    (h_in + 2 * padding.as_slice()[0] - 1 * (kernel.as_slice()[0] - 1) - 1)
                        / stride.as_slice()[0]
                        + 1,
                    (w_in + 2 * padding.as_slice()[1] - 1 * (kernel.as_slice()[1] - 1) - 1)
                        / stride.as_slice()[1]
                        + 1,
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
                let Attr::Shape(kernel) = &self.attrs[Self::MAXPOOL_ATTR_KERNEL];
                let Attr::Shape(stride) = &self.attrs[Self::MAXPOOL_ATTR_STRIDE];
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

impl From<Vec<usize>> for Attr {
    fn from(v: Vec<usize>) -> Self {
        Attr::Shape(Dimensions(v))
    }
}
