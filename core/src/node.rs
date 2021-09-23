use crate::{dim::Dimensions, tensor::Tensor};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

#[derive(Debug)]
pub enum Node {
    Conv2d(Conv2d),
    Add(Add),
    Relu(Relu),
    MaxPool(MaxPool),
    Reshape(Reshape),
    MatMul(MatMul),
    Tensor(Tensor),
    Input(Dimensions),
}

#[derive(Debug, Default)]
pub struct Conv2d {
    pub inputs: [Option<NodeId>; 2], // input, weight
    pub inputs_dims: [Dimensions; 2],
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub padding: Dimensions,
    pub output_dims: Dimensions,
}

#[derive(Debug, Default)]
pub struct Add {
    pub inputs: [Option<NodeId>; 2],
    pub inputs_dims: [Dimensions; 2],
    pub output_dims: Dimensions,
}

#[derive(Debug, Default)]
pub struct Relu {
    pub input: Option<NodeId>,
    pub input_dims: Dimensions,
}

#[derive(Debug, Default)]
pub struct MaxPool {
    pub input: Option<NodeId>,
    pub input_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub output_dims: Dimensions,
}

#[derive(Debug, Default)]
pub struct Reshape {
    pub input: Option<NodeId>,
    pub input_dims: Dimensions,
    pub output_dims: Dimensions,
}

#[derive(Debug, Default)]
pub struct MatMul {
    pub inputs: [Option<NodeId>; 2],
    pub inputs_dims: [Dimensions; 2],
    pub output_dims: Dimensions,
}

macro_rules! accessor {
    ($name:ident, $name_mut:ident, $tyname:ident) => {
        pub fn $name(&self) -> Option<&$tyname> {
            match self {
                Node::$tyname(node) => Some(node),
                _ => None,
            }
        }
        pub fn $name_mut(&mut self) -> Option<&mut $tyname> {
            match self {
                Node::$tyname(node) => Some(node),
                _ => None,
            }
        }
    };
}

impl Node {
    pub fn input(&self) -> &[Option<NodeId>] {
        use std::slice;
        match self {
            Self::Conv2d(n) => &n.inputs,
            Self::Add(n) => &n.inputs,
            Self::Relu(n) => slice::from_ref(&n.input),
            Self::MaxPool(n) => slice::from_ref(&n.input),
            Self::Reshape(n) => slice::from_ref(&n.input),
            Self::MatMul(n) => &n.inputs,
            Self::Tensor(_) => &[],
            Self::Input(_) => &[],
        }
    }

    pub fn input_dims(&self) -> &[Dimensions] {
        use std::slice;
        match self {
            Self::Conv2d(n) => &n.inputs_dims,
            Self::Add(n) => &n.inputs_dims,
            Self::Relu(n) => slice::from_ref(&n.input_dims),
            Self::MaxPool(n) => slice::from_ref(&n.input_dims),
            Self::Reshape(n) => slice::from_ref(&n.input_dims),
            Self::MatMul(n) => &n.inputs_dims,
            Self::Tensor(_) => &[],
            Self::Input(_) => &[],
        }
    }

    pub fn input_dims_mut(&mut self) -> &mut [Dimensions] {
        use std::slice;
        match self {
            Self::Conv2d(n) => &mut n.inputs_dims,
            Self::Add(n) => &mut n.inputs_dims,
            Self::Relu(n) => slice::from_mut(&mut n.input_dims),
            Self::MaxPool(n) => slice::from_mut(&mut n.input_dims),
            Self::Reshape(n) => slice::from_mut(&mut n.input_dims),
            Self::MatMul(n) => &mut n.inputs_dims,
            Self::Tensor(_) => &mut [],
            Self::Input(_) => &mut [],
        }
    }

    pub fn output_dims(&self) -> &Dimensions {
        match self {
            Self::Conv2d(n) => &n.output_dims,
            Self::Add(n) => &n.output_dims,
            Self::Relu(n) => n.output_dims(),
            Self::MaxPool(n) => &n.output_dims,
            Self::Reshape(n) => &n.output_dims,
            Self::MatMul(n) => &n.output_dims,
            Self::Tensor(n) => n.dims(),
            Self::Input(d) => d,
        }
    }

    pub fn compute_output_dims(&mut self) {
        match self {
            Self::Conv2d(n) => n.compute_output_dims(),
            Self::Add(n) => n.compute_output_dims(),
            Self::Relu(_) => {} // n.compute_output_dims(),
            Self::MaxPool(n) => n.compute_output_dims(),
            Self::Reshape(_) => {} // n.compute_output_dims(),
            Self::MatMul(n) => n.compute_output_dims(),
            Self::Tensor(_) => {} //n.compute_output_dims(),
            Self::Input(_) => {}  //n.compute_output_dims(),
        }
    }

    accessor!(as_conv2d, as_conv2d_mut, Conv2d);
    accessor!(as_add, as_add_mut, Add);
    accessor!(as_relu, as_relu_mut, Relu);
    accessor!(as_max_pool, as_max_pool_mut, MaxPool);
    accessor!(as_reshape, as_reshape_mut, Reshape);
    accessor!(as_mat_mul, as_mat_mul_mut, MatMul);
    accessor!(as_tensor, as_tensor_mut, Tensor);
    // accessor!(as_input, as_input_mut, Input);
}

impl Conv2d {
    pub fn new(input_dims: Dimensions, kernel: Dimensions) -> Self {
        Self {
            inputs_dims: [input_dims, Dimensions::default()],
            kernel,
            ..Conv2d::default()
        }
    }

    pub fn with_strides(mut self, strides: Dimensions) -> Self {
        self.stride = strides;
        self
    }

    pub fn with_padding(mut self, padding: Dimensions) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_input_node(mut self, input_node: NodeId) -> Self {
        self.inputs[0] = Some(input_node);
        self
    }

    pub fn with_weight_node(mut self, weight_node: NodeId, weight_dims: Dimensions) -> Self {
        self.inputs[1] = Some(weight_node);
        self.inputs_dims[1] = weight_dims;
        self
    }

    pub fn with_output_dims(mut self, output_dims: Dimensions) -> Self {
        self.output_dims = output_dims;
        self
    }

    pub fn compute_output_dims(&mut self) {
        let h_in = self.inputs_dims[0].as_slice()[2];
        let w_in = self.inputs_dims[0].as_slice()[3];
        self.output_dims = vec![
            self.inputs_dims[0].as_slice()[0],
            self.inputs_dims[1].as_slice()[0],
            (h_in + 2 * self.padding.as_slice()[0] - 1 * (self.kernel.as_slice()[0] - 1) - 1)
                / self.stride.as_slice()[0]
                + 1,
            (w_in + 2 * self.padding.as_slice()[1] - 1 * (self.kernel.as_slice()[1] - 1) - 1)
                / self.stride.as_slice()[1]
                + 1,
        ]
        .into();
    }
}

impl Add {
    pub fn compute_output_dims(&mut self) {
        let mut output_dims = vec![];
        let len = self.inputs_dims.iter().map(|d| d.len()).max().unwrap();
        for i in 0..len {
            let mut wanted_size = 1;
            for d in &self.inputs_dims {
                let len = d.len();
                let dim = if i < len {
                    d.as_slice()[len - i - 1]
                } else {
                    1
                };
                if dim != 1 {
                    if wanted_size != 1 && dim != wanted_size {
                        panic!();
                    }
                    wanted_size = dim;
                }
            }
            output_dims.push(wanted_size);
        }
        output_dims.reverse();
        self.output_dims = output_dims.into();
    }
}

impl Relu {
    pub fn new(input_dims: Dimensions) -> Self {
        Self {
            input_dims,
            ..Relu::default()
        }
    }

    pub fn with_input_node(mut self, input_node: NodeId) -> Self {
        self.input = Some(input_node);
        self
    }

    pub fn output_dims(&self) -> &Dimensions {
        &self.input_dims
    }
}

impl MaxPool {
    pub fn compute_output_dims(&mut self) {
        let h_in = self.input_dims.as_slice()[2];
        let w_in = self.input_dims.as_slice()[3];
        self.output_dims = vec![
            self.input_dims.as_slice()[0],
            self.input_dims.as_slice()[1],
            (h_in + 2 * 0 - 1 * (self.kernel.as_slice()[0] - 1) - 1) / self.stride.as_slice()[0]
                + 1,
            (w_in + 2 * 0 - 1 * (self.kernel.as_slice()[1] - 1) - 1) / self.stride.as_slice()[1]
                + 1,
        ]
        .into();
    }
}

impl MatMul {
    pub fn compute_output_dims(&mut self) {
        assert!(self.inputs_dims.iter().all(|d| d.len() == 2));
        self.output_dims = vec![
            self.inputs_dims[0].as_slice()[0],
            self.inputs_dims[1].as_slice()[1],
        ]
        .into();
    }
}

impl From<Conv2d> for Node {
    fn from(n: Conv2d) -> Node {
        Node::Conv2d(n)
    }
}

impl From<Add> for Node {
    fn from(n: Add) -> Node {
        Node::Add(n)
    }
}

impl From<Relu> for Node {
    fn from(n: Relu) -> Node {
        Node::Relu(n)
    }
}

impl From<MaxPool> for Node {
    fn from(n: MaxPool) -> Node {
        Node::MaxPool(n)
    }
}

impl From<Reshape> for Node {
    fn from(n: Reshape) -> Node {
        Node::Reshape(n)
    }
}

impl From<MatMul> for Node {
    fn from(n: MatMul) -> Node {
        Node::MatMul(n)
    }
}

impl From<Tensor> for Node {
    fn from(n: Tensor) -> Node {
        Node::Tensor(n)
    }
}

pub trait NodeBuilder {
    fn arena(&self) -> &NodeArena;
    fn arena_mut(&mut self) -> &mut NodeArena;

    fn new(&mut self, node: Node) -> NodeId {
        self.arena_mut().alloc(node)
    }

    fn new_relu(&mut self, input_node_id: NodeId) -> NodeId {
        let input_node = &self.arena()[input_node_id];
        let relu = Relu::new(input_node.output_dims().clone()).with_input_node(input_node_id);
        self.new(relu.into())
    }

    fn new_conv2d(
        &mut self,
        input_node_id: NodeId,
        weight_node_id: NodeId,
        kernel: Dimensions,
        strides: Dimensions,
        padding: Dimensions,
    ) -> NodeId {
        let input_node = &self.arena()[input_node_id];
        let weight_node = &self.arena()[weight_node_id];
        let mut conv2d = Conv2d::new(input_node.output_dims().clone(), kernel)
            .with_input_node(input_node_id)
            .with_weight_node(weight_node_id, weight_node.output_dims().clone())
            .with_strides(strides)
            .with_padding(padding);
        conv2d.compute_output_dims();
        self.new(conv2d.into())
    }
}
