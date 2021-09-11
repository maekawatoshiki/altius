use altius_core::{
    model::Model,
    node::{Add, Conv2d, MatMul, MaxPool, Node, NodeBuilder, NodeId, Relu, Reshape},
    tensor::Tensor,
};

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
            Node::Input => self.input.clone(),
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
                                        sum += weight.at(&[d, fd, fx as usize, fy as usize])
                                            * input.at(&[
                                                n,
                                                g * in_c_per_g + fd,
                                                ox as usize,
                                                oy as usize,
                                            ]);
                                    }
                                }
                            }
                            *output.at_mut(&[n, d, ax, ay]) = sum;
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
        let input = self.run_node(node.input_node.unwrap());

        let mut output = Tensor::new(node.output_dims.clone());
        for (i, v) in input.data().into_iter().enumerate() {
            output.data_mut()[i] = v.max(0.0);
        }
        output
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

                                let val = input.at(&[n, z, ox as usize, oy as usize]);

                                if val >= max {
                                    max = val;
                                }
                            }
                        }
                        *output.at_mut(&[n, z, ax, ay]) = if max == f32::MIN { 0.0 } else { max };
                        y += node.stride.as_slice()[1] as isize
                    }
                    x += node.stride.as_slice()[0] as isize
                }
            }
        }
        output
    }

    fn run_node_reshape(&mut self, node: &Reshape) -> Tensor {
        let mut output = Tensor::new(node.output_dims.clone());
        *output.data_vec_mut() = self.run_node(node.input_node.unwrap()).data_vec().clone();
        output
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
                    t += input_a.at(&[i, k]) * input_b.at(&[k, j]);
                }
                *output.at_mut(&[i, j]) = t;
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
                            *output.at_mut(&[n, z, x, y]) =
                                input_a.at(&[n, z, x, y]) + input_b.at(&[z, 0, 0]);
                        }
                    }
                }
            }
            return output;
        }

        todo!()
    }
}

#[test]
fn run() {
    use rayon::prelude::*;
    use std::cmp::Ordering;
    let mnist = mnist();

    let test = include_str!("../../core/examples/MNIST_test.txt");
    let test_lines: Vec<&str> = test.split("\n").collect();
    let mut inputs = vec![];
    for line in test_lines {
        if line.is_empty() {
            continue;
        }
        let nums: Vec<&str> = line.split(",").collect();
        let expected: i32 = nums[0].parse().unwrap();
        let pixels: Vec<f32> = nums[1..]
            .iter()
            .map(|s| s.parse::<f32>().unwrap() / 255.0)
            .collect();
        inputs.push((expected, pixels));
    }
    let correct: i32 = inputs
        .par_iter()
        .map(|(expected, input)| {
            let mut i = Interpreter::new(
                &mnist,
                Tensor::new(vec![1, 1, 28, 28].into()).with_data(input.clone().into()),
            );
            let v = i.run();
            let inferred = v
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            // println!("expected: {}, inferred: {}", expected, inferred);
            (*expected == inferred as i32) as i32
        })
        .sum();
    println!("accuracy: {}", correct as f32 / inputs.len() as f32);
}

#[cfg(test)]
fn mnist() -> Model {
    use altius_core::{node::*, tensor::*};
    let mut m = Model::new();
    let input = m.new(Node::Input);
    let conv_weight = m.new(
        Tensor::new(vec![8, 1, 5, 5].into())
            .with_data(
                include!("../../core/examples/conv1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let conv = m.new(
        Conv2d {
            input_dims: vec![1, 1, 28, 28].into(),
            weight_dims: vec![8, 1, 5, 5].into(),
            weight_node: Some(conv_weight),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            padding: vec![2, 2].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(input),
            ..Default::default()
        }
        .into(),
    );
    // m.input_node = Some(conv);
    let add_input_b = m.new(
        Tensor::new(vec![8, 1, 1].into())
            .with_data(
                include!("../../core/examples/add1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add = m.new(
        Add {
            input_a_dims: vec![1, 8, 28, 28].into(),
            input_b_dims: vec![8, 1, 1].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_a_node: Some(conv),
            input_b_node: Some(add_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu = m.new(
        Relu {
            input_dims: vec![1, 8, 28, 28].into(),
            output_dims: vec![1, 8, 28, 28].into(),
            input_node: Some(add),
            ..Default::default()
        }
        .into(),
    );
    let max_pool = m.new(
        MaxPool {
            input_dims: vec![1, 8, 28, 28].into(),
            kernel: vec![2, 2].into(),
            stride: vec![2, 2].into(),
            output_dims: vec![1, 8, 14, 14].into(),
            input_node: Some(relu),
            ..Default::default()
        }
        .into(),
    );
    let conv2_weight = m.new(
        Tensor::new(vec![16, 8, 5, 5].into())
            .with_data(
                include!("../../core/examples/conv2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let conv2 = m.new(
        Conv2d {
            input_dims: vec![1, 8, 14, 14].into(),
            weight_dims: vec![16, 8, 5, 5].into(),
            kernel: vec![5, 5].into(),
            stride: vec![1, 1].into(),
            padding: vec![2, 2].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_node: Some(max_pool),
            weight_node: Some(conv2_weight),
            ..Default::default()
        }
        .into(),
    );
    let add2_input_b = m.new(
        Tensor::new(vec![16, 1, 1].into())
            .with_data(
                include!("../../core/examples/add2")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add2 = m.new(
        Add {
            input_a_dims: vec![1, 16, 14, 14].into(),
            input_b_dims: vec![16, 1, 1].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_a_node: Some(conv2),
            input_b_node: Some(add2_input_b),
            ..Default::default()
        }
        .into(),
    );
    let relu2 = m.new(
        Relu {
            input_dims: vec![1, 16, 14, 14].into(),
            output_dims: vec![1, 16, 14, 14].into(),
            input_node: Some(add2),
            ..Default::default()
        }
        .into(),
    );
    let max_pool2 = m.new(
        MaxPool {
            input_dims: vec![1, 16, 14, 14].into(),
            kernel: vec![3, 3].into(),
            stride: vec![3, 3].into(),
            output_dims: vec![1, 16, 4, 4].into(),
            input_node: Some(relu2),
            ..Default::default()
        }
        .into(),
    );
    let reshape = m.new(
        Reshape {
            input_dims: vec![1, 16, 4, 4].into(),
            output_dims: vec![1, 256].into(),
            input_node: Some(max_pool2),
            ..Default::default()
        }
        .into(),
    );
    let reshape2_input = m.new(
        Tensor::new(vec![16, 4, 4, 10].into())
            .with_data(
                include!("../../core/examples/reshape1")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let reshape2 = m.new(
        Reshape {
            input_dims: vec![16, 4, 4, 10].into(),
            output_dims: vec![256, 10].into(),
            input_node: Some(reshape2_input),
            ..Default::default()
        }
        .into(),
    );
    let mat_mal = m.new(
        MatMul {
            input_a_dims: vec![1, 256].into(),
            input_b_dims: vec![256, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(reshape),
            input_b_node: Some(reshape2),
            ..Default::default()
        }
        .into(),
    );
    let add3_input_b = m.new(
        Tensor::new(vec![1, 10].into())
            .with_data(
                include!("../../core/examples/add3")
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into(),
    );
    let add3 = m.new(
        Add {
            input_a_dims: vec![1, 10].into(),
            input_b_dims: vec![1, 10].into(),
            output_dims: vec![1, 10].into(),
            input_a_node: Some(mat_mal),
            input_b_node: Some(add3_input_b),
            ..Default::default()
        }
        .into(),
    );
    m.output_node = Some(add3);
    m
}
