use altius_core::{
    model::{Model, Model2},
    node::{
        Add, Attr, Conv2d, MatMul, MaxPool, Node, Node2, Node2Id, NodeBuilder, NodeId, Op, Relu,
        Reshape,
    },
    tensor::{Tensor, Tensor2},
    value::ValueId,
};
use rustc_hash::FxHashMap;

pub struct Interpreter2<'a> {
    model: &'a Model2,
    values: FxHashMap<ValueId, Tensor2>,
}

impl<'a> Interpreter2<'a> {
    pub fn new(model: &'a Model2) -> Self {
        Interpreter2 {
            model,
            values: FxHashMap::default(),
        }
    }

    pub fn run(&mut self, input: Tensor2) -> &Tensor2 {
        assert!(self.model.inputs.len() == 1);
        assert!(self.model.outputs.len() == 1);

        // Set input & initializer values.
        self.values.insert(self.model.inputs[0], input);
        self.values.extend(self.model.inits.clone().into_iter());

        let nodes = self.model.topo_sort_nodes();

        for node in nodes {
            self.run_node(node);
        }

        &self.values[&self.model.outputs[0]]
    }

    fn run_node(&mut self, node: Node2Id) {
        let node = &self.model.nodes[node];
        let mut inputs = vec![];
        for input in node.inputs.iter() {
            inputs.push(self.values[input].clone());
        }
        let output_shapes = node.compute_output_shapes(&inputs);
        let mut outputs = vec![];
        for output_shape in output_shapes {
            outputs.push(Tensor2::new(output_shape));
        }

        // TODO: Actual kernel runs here!
        match node.op {
            Op::Conv2d => self.run_node_conv2d(node, &inputs, &mut outputs),
            Op::Add => self.run_node_add(node, &inputs, &mut outputs),
            Op::MaxPool => self.run_node_max_pool(node, &inputs, &mut outputs),
            Op::Reshape => self.run_node_reshape(node, &inputs, &mut outputs),
            Op::MatMul => self.run_node_mat_mul(node, &inputs, &mut outputs),
            Op::ReLU => self.run_node_relu(node, &inputs, &mut outputs),
        }

        // println!("{:?} {:?}", node.op, outputs);

        for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
            self.values.insert(val, output);
        }
    }

    fn run_node_conv2d(&mut self, node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input = &inputs[Node2::CONV2D_IN];
        let weight = &inputs[Node2::CONV2D_WEIGHT];
        let output = &mut outputs[0];

        let Attr::Shape(kernel) = &node.attrs[Node2::CONV2D_ATTR_KERNEL];
        let Attr::Shape(padding) = &node.attrs[Node2::CONV2D_ATTR_PADDING];
        let Attr::Shape(stride) = &node.attrs[Node2::CONV2D_ATTR_STRIDE];

        let dilation = 1;
        let group = 1;
        let in_c_per_g = input.dims().as_slice()[1] / group;
        let out_c_per_g = output.dims().as_slice()[1] / group;

        // let mut output = Tensor::new(node.output_dims.clone());
        for n in 0..input.dims().as_slice()[0] {
            for g in 0..group {
                for d in (g * out_c_per_g)..((g + 1) * out_c_per_g) {
                    let mut x = -(padding.as_slice()[0] as isize);
                    for ax in 0..output.dims().as_slice()[2] {
                        let mut y = -(padding.as_slice()[0] as isize);
                        for ay in 0..output.dims().as_slice()[3] {
                            let mut sum = 0.0;
                            for fx in 0..kernel.as_slice()[0] as isize {
                                for fy in 0..kernel.as_slice()[1] as isize {
                                    let ox = x + fx * dilation;
                                    let oy = y + fy * dilation;

                                    if ox < 0
                                        || oy < 0
                                        || ox >= input.dims().as_slice()[2] as isize
                                        || oy >= input.dims().as_slice()[3] as isize
                                    {
                                        continue;
                                    }

                                    for fd in 0..in_c_per_g {
                                        sum += weight.at_4d(d, fd, fx as usize, fy as usize)
                                            * input.at_4d(
                                                n,
                                                g * in_c_per_g + fd,
                                                ox as usize,
                                                oy as usize,
                                            );
                                    }
                                }
                            }
                            *output.at_4d_mut(n, d, ax, ay) = sum;
                            y += stride.as_slice()[1] as isize
                        }
                        x += stride.as_slice()[0] as isize
                    }
                }
            }
        }
    }

    fn run_node_max_pool(&mut self, node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input = &inputs[Node2::MAXPOOL_IN];
        let output = &mut outputs[Node2::MAXPOOL_OUT];

        let Attr::Shape(kernel) = &node.attrs[Node2::MAXPOOL_ATTR_KERNEL];
        let Attr::Shape(stride) = &node.attrs[Node2::MAXPOOL_ATTR_STRIDE];

        assert!(input.dims().len() == 4);
        assert!(output.dims().len() == 4);

        for n in 0..output.dims().as_slice()[0] {
            for z in 0..input.dims().as_slice()[1] {
                let mut x = 0isize; // TODO: pad
                for ax in 0..output.dims().as_slice()[2] {
                    let mut y = 0isize; // TODO: pad
                    for ay in 0..output.dims().as_slice()[3] {
                        let mut max = f32::MIN;
                        for fx in 0..kernel.as_slice()[0] as isize {
                            for fy in 0..kernel.as_slice()[1] as isize {
                                let ox = x + fx;
                                let oy = y + fy;

                                if ox < 0
                                    || oy < 0
                                    || ox >= input.dims().as_slice()[2] as isize
                                    || oy >= input.dims().as_slice()[3] as isize
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
                        y += stride.as_slice()[1] as isize
                    }
                    x += stride.as_slice()[0] as isize
                }
            }
        }
    }

    fn run_node_add(&mut self, _node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input_a = &inputs[Node2::ADD_IN_A];
        let input_b = &inputs[Node2::ADD_IN_B];
        let output = &mut outputs[Node2::ADD_OUT];

        if input_a.dims() == input_b.dims() {
            for (i, (a, b)) in input_a
                .data()
                .as_f32()
                .unwrap()
                .iter()
                .zip(input_b.data().as_f32().unwrap().iter())
                .enumerate()
            {
                output.data_mut().as_f32_mut().unwrap()[i] = a + b;
            }

            return;
        }

        if input_a.dims().len() == 4 && input_b.dims().len() == 3 {
            assert!(input_a.dims().as_slice()[1] == input_b.dims().as_slice()[0]);
            assert!(input_b.dims().as_slice()[1] == 1);
            assert!(input_b.dims().as_slice()[2] == 1);

            for n in 0..input_a.dims().as_slice()[0] {
                for z in 0..input_a.dims().as_slice()[1] {
                    for x in 0..input_a.dims().as_slice()[2] {
                        for y in 0..input_a.dims().as_slice()[3] {
                            *output.at_4d_mut(n, z, x, y) =
                                input_a.at_4d(n, z, x, y) + input_b.at_3d(z, 0, 0);
                        }
                    }
                }
            }

            return;
        }
    }

    fn run_node_mat_mul(&mut self, _node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input_a = &inputs[Node2::MATMUL_IN_A];
        let input_b = &inputs[Node2::MATMUL_IN_B];
        let output = &mut outputs[Node2::MATMUL_OUT];

        assert!(input_a.dims().len() == 2);
        assert!(input_b.dims().len() == 2);
        assert!(input_a.dims().as_slice()[1] == input_b.dims().as_slice()[0]);

        for i in 0..input_a.dims().as_slice()[0] {
            for j in 0..input_b.dims().as_slice()[1] {
                let mut t = 0.0;
                for k in 0..input_b.dims().as_slice()[0] {
                    t += input_a.at_2d(i, k) * input_b.at_2d(k, j);
                }
                *output.at_2d_mut(i, j) = t;
            }
        }
    }

    fn run_node_relu(&mut self, _node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input = &inputs[Node2::RELU_IN];
        let output = &mut outputs[Node2::RELU_OUT];

        for (i, o) in input
            .data()
            .as_f32()
            .unwrap()
            .iter()
            .zip(output.data_mut().as_f32_mut().unwrap().iter_mut())
        {
            *o = i.max(0.0);
        }
    }

    fn run_node_reshape(&mut self, _node: &Node2, inputs: &[Tensor2], outputs: &mut [Tensor2]) {
        let input = &inputs[Node2::RESHAPE_IN];
        let shape = &inputs[Node2::RESHAPE_SHAPE];
        let output = &mut outputs[Node2::RESHAPE_OUT];
        *output = input.clone().reshape_into(
            shape
                .data()
                .as_i64()
                .unwrap()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<_>>()
                .into(),
        );
    }
}

pub struct Interpreter<'a> {
    model: &'a Model,
    input: Tensor,
}

impl<'a> Interpreter<'a> {
    pub fn new(model: &'a Model, input: Tensor) -> Self {
        Self { model, input }
    }

    pub fn run(&mut self) -> Tensor {
        self.run_node(self.model.output_node.unwrap())
    }

    pub fn run_node(&mut self, id: NodeId) -> Tensor {
        let node = &self.model.arena()[id];
        match node {
            Node::Input(_) => self.input.clone(),
            Node::Tensor(tensor) => tensor.clone(),
            Node::Conv2d(node) => self.run_node_conv2d(node),
            Node::Relu(node) => self.run_node_relu(node),
            Node::MaxPool(node) => self.run_node_max_pool(node),
            Node::Reshape(node) => self.run_node_reshape(node),
            Node::MatMul(node) => self.run_node_mat_mul(node),
            Node::Add(node) => self.run_node_add(node),
        }
    }

    fn run_node_conv2d(&mut self, node: &Conv2d) -> Tensor {
        let input = self.run_node(node.input_node.unwrap());
        let weight = self.run_node(node.weight_node.unwrap());

        let dilation = 1;
        let group = 1;
        let in_c_per_g = node.input_dims.as_slice()[1] / group;
        let out_c_per_g = node.output_dims.as_slice()[1] / group;

        let mut output = Tensor::new(node.output_dims.clone());
        for n in 0..node.input_dims.as_slice()[0] {
            for g in 0..group {
                for d in (g * out_c_per_g)..((g + 1) * out_c_per_g) {
                    let mut x = -(node.padding.as_slice()[0] as isize);
                    for ax in 0..node.output_dims.as_slice()[2] {
                        let mut y = -(node.padding.as_slice()[0] as isize);
                        for ay in 0..node.output_dims.as_slice()[3] {
                            let mut sum = 0.0;
                            for fx in 0..node.kernel.as_slice()[0] as isize {
                                for fy in 0..node.kernel.as_slice()[1] as isize {
                                    let ox = x + fx * dilation;
                                    let oy = y + fy * dilation;

                                    if ox < 0
                                        || oy < 0
                                        || ox >= node.input_dims.as_slice()[2] as isize
                                        || oy >= node.input_dims.as_slice()[3] as isize
                                    {
                                        continue;
                                    }

                                    for fd in 0..in_c_per_g {
                                        sum += weight.at_4d(d, fd, fx as usize, fy as usize)
                                            * input.at_4d(
                                                n,
                                                g * in_c_per_g + fd,
                                                ox as usize,
                                                oy as usize,
                                            );
                                    }
                                }
                            }
                            *output.at_4d_mut(n, d, ax, ay) = sum;
                            y += node.stride.as_slice()[1] as isize
                        }
                        x += node.stride.as_slice()[0] as isize
                    }
                }
            }
        }
        output
    }

    fn run_node_relu(&mut self, node: &Relu) -> Tensor {
        let mut t = self.run_node(node.input_node.unwrap());
        for v in t.data_mut() {
            *v = v.max(0.0);
        }
        t
    }

    fn run_node_max_pool(&mut self, node: &MaxPool) -> Tensor {
        let input = self.run_node(node.input_node.unwrap());

        assert!(node.input_dims.len() == 4);
        assert!(node.output_dims.len() == 4);

        let mut output = Tensor::new(node.output_dims.clone());
        for n in 0..node.output_dims.as_slice()[0] {
            for z in 0..node.input_dims.as_slice()[1] {
                let mut x = 0isize; // TODO: pad
                for ax in 0..node.output_dims.as_slice()[2] {
                    let mut y = 0isize; // TODO: pad
                    for ay in 0..node.output_dims.as_slice()[3] {
                        let mut max = f32::MIN;
                        for fx in 0..node.kernel.as_slice()[0] as isize {
                            for fy in 0..node.kernel.as_slice()[1] as isize {
                                let ox = x + fx;
                                let oy = y + fy;

                                if ox < 0
                                    || oy < 0
                                    || ox >= node.input_dims.as_slice()[2] as isize
                                    || oy >= node.input_dims.as_slice()[3] as isize
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
                        y += node.stride.as_slice()[1] as isize
                    }
                    x += node.stride.as_slice()[0] as isize
                }
            }
        }
        output
    }

    fn run_node_reshape(&mut self, node: &Reshape) -> Tensor {
        let input = self.run_node(node.input_node.unwrap());
        input.reshape_into(node.output_dims.clone())
    }

    fn run_node_mat_mul(&mut self, node: &MatMul) -> Tensor {
        let input_a = self.run_node(node.input_a_node.unwrap());
        let input_b = self.run_node(node.input_b_node.unwrap());

        assert!(node.input_a_dims.len() == 2);
        assert!(node.input_b_dims.len() == 2);
        assert!(node.input_a_dims.as_slice()[1] == node.input_b_dims.as_slice()[0]);

        let mut output = Tensor::new(node.output_dims.clone());
        for i in 0..input_a.dims().as_slice()[0] {
            for j in 0..input_b.dims().as_slice()[1] {
                let mut t = 0.0;
                for k in 0..input_b.dims().as_slice()[0] {
                    t += input_a.at_2d(i, k) * input_b.at_2d(k, j);
                }
                *output.at_2d_mut(i, j) = t;
            }
        }
        output
    }

    fn run_node_add(&mut self, node: &Add) -> Tensor {
        let input_a = self.run_node(node.input_a_node.unwrap());
        let input_b = self.run_node(node.input_b_node.unwrap());

        if node.input_a_dims == node.input_b_dims {
            let mut output = Tensor::new(node.output_dims.clone());
            for (i, (a, b)) in input_a.data().iter().zip(input_b.data().iter()).enumerate() {
                output.data_mut()[i] = a + b;
            }
            return output;
        }

        if node.input_a_dims.len() == 4 && node.input_b_dims.len() == 3 {
            assert!(node.input_a_dims.as_slice()[1] == node.input_b_dims.as_slice()[0]);
            assert!(node.input_b_dims.as_slice()[1] == 1);
            assert!(node.input_b_dims.as_slice()[2] == 1);

            let mut output = Tensor::new(node.output_dims.clone());
            for n in 0..node.input_a_dims.as_slice()[0] {
                for z in 0..node.input_a_dims.as_slice()[1] {
                    for x in 0..node.input_a_dims.as_slice()[2] {
                        for y in 0..node.input_a_dims.as_slice()[3] {
                            *output.at_4d_mut(n, z, x, y) =
                                input_a.at_4d(n, z, x, y) + input_b.at_3d(z, 0, 0);
                        }
                    }
                }
            }
            return output;
        }

        todo!()
    }
}
