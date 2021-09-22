use crate::{dim::Dimensions, tensor::Tensor};
use id_arena::{Arena, Id};

pub type NodeId = Id<Node>;
pub type NodeArena = Arena<Node>;

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

#[derive(Default)]
pub struct Conv2d {
    pub input_dims: Dimensions,
    pub weight_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub padding: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
    pub weight_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Add {
    pub input_a_dims: Dimensions,
    pub input_b_dims: Dimensions,
    pub input_b: Tensor,
    pub output_dims: Dimensions,
    pub input_a_node: Option<NodeId>,
    pub input_b_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Relu {
    pub input_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct MaxPool {
    pub input_dims: Dimensions,
    pub kernel: Dimensions,
    pub stride: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct Reshape {
    pub input_dims: Dimensions,
    pub output_dims: Dimensions,
    pub input_node: Option<NodeId>,
}

#[derive(Default)]
pub struct MatMul {
    pub input_a_dims: Dimensions,
    pub input_b_dims: Dimensions,
    pub output_dims: Dimensions,
    pub input_a_node: Option<NodeId>,
    pub input_b_node: Option<NodeId>,
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
    pub fn output_dims(&self) -> &Dimensions {
        match self {
            Self::Conv2d(n) => &n.output_dims,
            Self::Add(n) => &n.output_dims,
            Self::Relu(n) => n.output_dims(),
            Self::MaxPool(n) => &n.output_dims,
            Self::Reshape(n) => &n.output_dims,
            Self::MatMul(n) => &n.output_dims,
            Self::Tensor(n) => &n.dims(),
            Self::Input(d) => &d,
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
            input_dims,
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
        self.input_node = Some(input_node);
        self
    }

    pub fn with_weight_node(mut self, weight_node: NodeId, weight_dims: Dimensions) -> Self {
        self.weight_node = Some(weight_node);
        self.weight_dims = weight_dims;
        self
    }

    pub fn with_output_dims(mut self, output_dims: Dimensions) -> Self {
        self.output_dims = output_dims;
        self
    }

    pub fn calc_output_dims(
        input: &Dimensions,
        weight: &Dimensions,
        kernel: &Dimensions,
        strides: &Dimensions,
        padding: &Dimensions,
    ) -> Dimensions {
        let h_in = input.as_slice()[2];
        let w_in = input.as_slice()[3];
        vec![
            input.as_slice()[0],
            weight.as_slice()[0],
            (h_in + 2 * padding.as_slice()[0] - 1 * (kernel.as_slice()[0] - 1) - 1)
                / strides.as_slice()[0]
                + 1,
            (w_in + 2 * padding.as_slice()[1] - 1 * (kernel.as_slice()[1] - 1) - 1)
                / strides.as_slice()[1]
                + 1,
        ]
        .into()
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
        self.input_node = Some(input_node);
        self
    }

    pub fn output_dims(&self) -> &Dimensions {
        &self.input_dims
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
        let output_dims = Conv2d::calc_output_dims(
            input_node.output_dims(),
            weight_node.output_dims(),
            &kernel,
            &strides,
            &padding,
        );
        let conv2d = Conv2d::new(input_node.output_dims().clone(), kernel)
            .with_input_node(input_node_id)
            .with_weight_node(weight_node_id, weight_node.output_dims().clone())
            .with_strides(strides)
            .with_padding(padding)
            .with_output_dims(output_dims.into());
        self.new(conv2d.into())
    }
}
