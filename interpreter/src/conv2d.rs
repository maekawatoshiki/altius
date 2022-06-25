use altius_core::{
    node::{Conv2d, Node},
    tensor::Tensor,
};
#[cfg(feature = "cuda")]
use cudnn::{
    ActivationDescriptor, ActivationMode, ConvDescriptor, ConvFwdAlgo, ConvMode, CudnnContext,
    FilterDescriptor, NanPropagation, ScalarC, TensorDescriptor,
};
#[cfg(feature = "cuda")]
use cust::memory::DeviceBuffer;

pub struct Conv2dCtx<'a> {
    #[cfg(feature = "cuda")]
    pub cudnn: &'a CudnnContext,
    pub op: &'a Conv2d,
    pub inputs: &'a [Tensor],
    pub outputs: &'a mut [Tensor],
}

#[cfg(not(feature = "cuda"))]
pub fn run(ctx: &mut Conv2dCtx) {
    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let bias = ctx.inputs.get(Node::CONV2D_BIAS).map_or(
        Tensor::zeros::<f32>(vec![weight.dims()[0]].into()),
        Clone::clone,
    );
    let output = &mut ctx.outputs[0];

    let kernel = &ctx.op.kernel_shape;
    let padding = &ctx.op.padding;
    let stride = &ctx.op.strides;

    let dilation = 1;
    let group = ctx.op.group as usize;
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

#[cfg(feature = "cuda")]
pub fn run(ctx: &mut Conv2dCtx) {
    let conv = ctx.op;
    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let bias = ctx.inputs.get(Node::CONV2D_BIAS).map_or(
        Tensor::zeros::<f32>(vec![weight.dims()[0]].into()),
        Clone::clone,
    );
    let output = &mut ctx.outputs[0];

    let padding = [conv.padding[0] as i32, conv.padding[1] as i32];
    let stride = [conv.strides[0] as i32, conv.strides[1] as i32];
    let dilation = [1, 1];
    let mode = ConvMode::CrossCorrelation;

    let conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode).unwrap();

    let input_cuda = DeviceBuffer::from_slice(input.data::<f32>()).unwrap();
    let weight_cuda = DeviceBuffer::from_slice(weight.data::<f32>()).unwrap();
    let bias_cuda = DeviceBuffer::from_slice(bias.data::<f32>()).unwrap();
    let mut output_cuda = DeviceBuffer::from_slice(output.data::<f32>()).unwrap();
    let input_desc =
        TensorDescriptor::<f32>::new_format(&input.dims().to_i32_vec(), ScalarC::Nchw).unwrap();
    let weight_desc =
        FilterDescriptor::<f32>::new(&weight.dims().to_i32_vec(), ScalarC::Nchw).unwrap();
    let bias_desc =
        TensorDescriptor::<f32>::new_format(&[1i32, bias.dims()[0] as i32, 1, 1], ScalarC::Nchw)
            .unwrap();
    let output_desc =
        TensorDescriptor::<f32>::new_format(&output.dims().to_i32_vec(), ScalarC::Nchw).unwrap();

    let algo = ConvFwdAlgo::Gemm;

    let size = ctx
        .cudnn
        .get_convolution_forward_workspace_size(
            &input_desc,
            &weight_desc,
            &output_desc,
            &conv_desc,
            algo,
        )
        .unwrap();

    let mut workspace =
        size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });

    let alpha = 1.;
    let beta = 0.;

    let mode = ActivationMode::Identity;
    let nan_opt = NanPropagation::NotPropagateNaN;
    let coefficient = None;

    let activation_desc = ActivationDescriptor::new(mode, nan_opt, coefficient).unwrap();

    let z = DeviceBuffer::from_slice(&[0.0f32]).unwrap();
    let z_desc = TensorDescriptor::<f32>::new_format(&[1, 1, 1, 1], ScalarC::Nchw).unwrap();

    ctx.cudnn
        .convolution_bias_act_forward(
            alpha,
            &input_desc,
            &input_cuda,
            &weight_desc,
            &weight_cuda,
            &conv_desc,
            algo,
            workspace.as_mut(),
            beta,
            &z_desc,
            &z,
            &bias_desc,
            &bias_cuda,
            &activation_desc,
            &output_desc,
            &mut output_cuda,
        )
        .unwrap();

    *output = Tensor::new(output.dims().clone(), output_cuda.as_host_vec().unwrap());
}
