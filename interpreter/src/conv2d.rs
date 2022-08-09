use altius_core::{
    node::{Conv2d, Node},
    tensor::Tensor,
};

#[cfg(feature = "cuda")]
use crate::SafeCudnnContext;
#[cfg(feature = "cuda")]
pub use cudnn::{
    ActivationDescriptor, ActivationMode, ConvDescriptor, ConvFwdAlgo, ConvMode, FilterDescriptor,
    NanPropagation, ScalarC, TensorDescriptor,
};
#[cfg(feature = "cuda")]
pub use cust::memory::DeviceBuffer;

pub struct Conv2dCtx<'a> {
    #[cfg(feature = "cuda")]
    pub cudnn: &'a SafeCudnnContext,
    pub op: &'a Conv2d,
    pub inputs: &'a [Tensor],
    pub outputs: &'a mut [Tensor],
}

#[cfg(not(feature = "cuda"))]
pub fn compute(ctx: &mut Conv2dCtx) {
    use ndarray::{linalg, s, Array4, Array6, ArrayView3, ArrayView4};

    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let output = &mut ctx.outputs[0];

    let kernel = &ctx.op.kernel_shape;
    let padding = &ctx.op.padding;
    let stride = &ctx.op.strides;

    let batch_size = input.dims()[0];
    let input_c = input.dims()[1];
    let input_h = input.dims()[2];
    let input_w = input.dims()[3];
    let output_h = output.dims()[2];
    let output_w = output.dims()[3];
    let _dilation = 1;
    let group = ctx.op.group as usize;
    let in_c_per_g = input_c / group;
    let out_c_per_g = output.dims()[1] / group;

    assert!(padding.len() == 4);
    let pad_t = padding[0];
    let pad_l = padding[1];
    let _pad_b = padding[2];
    let _pad_r = padding[3];

    let mut input_ = Array4::zeros([
        batch_size,
        input_c,
        input_h + pad_t * 2,
        input_w + pad_l * 2,
    ]);
    input_
        .slice_mut(s![.., .., pad_t..input_h + pad_t, pad_l..input_w + pad_l])
        .assign(&ArrayView4::from_shape(input.fixed_dims::<4>(), input.data::<f32>()).unwrap());
    let weight_ = ArrayView3::from_shape(
        [group, out_c_per_g, in_c_per_g * kernel[0] * kernel[1]],
        weight.data::<f32>(),
    )
    .unwrap();
    let mut output_: Array4<f32> = ctx.inputs.get(Node::CONV2D_BIAS).map_or_else(
        || Array4::zeros([batch_size, group, out_c_per_g, output_h * output_w]),
        |bias| {
            ArrayView4::from_shape([1, group, out_c_per_g, 1], bias.data::<f32>())
                .unwrap()
                .broadcast([batch_size, group, out_c_per_g, output_h * output_w])
                .unwrap()
                .to_owned()
        },
    );

    let mut col = Array6::<f32>::zeros([
        batch_size, input_c, kernel[0], kernel[1], output_h, output_w,
    ]);
    // TODO: The following code gets slower when rewriting it without ndarray.
    //       Thus, ndarray is the best solution so far.
    for fy in 0..kernel[0] {
        let fy_max = fy + stride[0] * output_h;
        for fx in 0..kernel[1] {
            let fx_max = fx + stride[1] * output_w;
            col.slice_mut(s![.., .., fy, fx, .., ..])
                .assign(&input_.slice(s![.., .., fy..fy_max;stride[0], fx..fx_max;stride[1]]))
        }
    }
    let col = col
        .into_shape([
            batch_size,
            group,
            in_c_per_g * kernel[0] * kernel[1],
            output_h * output_w,
        ])
        .unwrap();

    for n in 0..batch_size {
        for g in 0..group {
            linalg::general_mat_mul(
                1.0,
                &weight_.slice(s![g, .., ..]),
                &col.slice(s![n, g, .., ..]),
                1.0,
                &mut output_.slice_mut(s![n, g, .., ..]),
            );
        }
    }

    output.set_raw_vec(output_.into_raw_vec())
}

#[cfg(feature = "cuda")]
pub fn compute(ctx: &mut Conv2dCtx) {
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
        .0
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
        .0
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

    output.set_raw_vec(output_cuda.as_host_vec().unwrap());
}
