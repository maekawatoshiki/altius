use altius_core::{
    node::{Conv2d, Node},
    tensor::Tensor,
};

pub fn run(conv: &Conv2d, inputs: &[Tensor], outputs: &mut [Tensor]) {
    let input = &inputs[Node::CONV2D_IN];
    let weight = &inputs[Node::CONV2D_WEIGHT];
    let bias = inputs.get(Node::CONV2D_BIAS).map_or(
        Tensor::zeros::<f32>(vec![weight.dims()[0]].into()),
        Clone::clone,
    );
    let output = &mut outputs[0];

    let kernel = &conv.kernel_shape;
    let padding = &conv.padding;
    let stride = &conv.strides;

    let dilation = 1;
    let group = conv.group as usize;
    let in_c_per_g = input.dims()[1] / group;
    let out_c_per_g = output.dims()[1] / group;

    // let mut output = Tensor::new(node.output_dims.clone());
    for n in 0..input.dims()[0] {
        for g in 0..group {
            for d in (g * out_c_per_g)..((g + 1) * out_c_per_g) {
                let mut x = -(padding[0] as isize);
                for ax in 0..output.dims()[2] {
                    let mut y = -(padding[1] as isize);
                    for ay in 0..output.dims()[3] {
                        let mut sum = bias.at(&[d]);
                        for fx in 0..kernel[0] as isize {
                            for fy in 0..kernel[1] as isize {
                                let ox = x + fx * dilation;
                                let oy = y + fy * dilation;

                                if ox < 0
                                    || oy < 0
                                    || ox >= input.dims()[2] as isize
                                    || oy >= input.dims()[3] as isize
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
                        y += stride[1] as isize
                    }
                    x += stride[0] as isize
                }
            }
        }
    }
}
