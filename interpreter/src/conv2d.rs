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
    use ndarray::{linalg, s, Array3, Array4, Array6, ArrayView3, ArrayView4};

    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let output = &mut ctx.outputs[0];

    let kernel = &ctx.op.kernel_shape;
    let padding = &ctx.op.padding;
    let stride = &ctx.op.strides;

    let _dilation = 1;
    let group = ctx.op.group as usize;
    let in_c_per_g = input.dims()[1] / group;
    let out_c_per_g = output.dims()[1] / group;

    assert!(
        padding.len() == 2
            || (padding.len() == 4 && padding[0] == padding[2] && padding[1] == padding[3])
    );

    let mut input_ = Array4::zeros([
        input.dims()[0],
        input.dims()[1],
        input.dims()[2] + padding[0] * 2,
        input.dims()[3] + padding[1] * 2,
    ]);
    input_
        .slice_mut(s![
            ..,
            ..,
            padding[0]..input.dims()[2] + padding[0],
            padding[1]..input.dims()[3] + padding[1]
        ])
        .assign(
            &ArrayView4::from_shape(
                [
                    input.dims()[0],
                    input.dims()[1],
                    input.dims()[2],
                    input.dims()[3],
                ],
                input.data::<f32>(),
            )
            .unwrap(),
        );
    let weight_ = ArrayView3::from_shape(
        [
            group,
            out_c_per_g,
            in_c_per_g * weight.dims()[2] * weight.dims()[3],
        ],
        weight.data::<f32>(),
    )
    .unwrap();
    let mut output_ = ctx.inputs.get(Node::CONV2D_BIAS).map_or_else(
        || {
            Array3::zeros([
                group,
                out_c_per_g,
                input.dims()[0] * output.dims()[2] * output.dims()[3],
            ])
        },
        |bias| {
            ArrayView3::from_shape([group, out_c_per_g, 1], bias.data::<f32>())
                .unwrap()
                .broadcast([
                    group,
                    out_c_per_g,
                    input.dims()[0] * output.dims()[2] * output.dims()[3],
                ])
                .unwrap()
                .to_owned()
        },
    );

    let mut col = Array6::<f32>::zeros([
        input.dims()[0],
        input.dims()[1],
        weight.dims()[2],
        weight.dims()[3],
        output.dims()[2],
        output.dims()[3],
    ]);

    for fy in 0..kernel[0] {
        let fy_max = fy + stride[0] * output.dims()[2];
        for fx in 0..kernel[1] {
            let fx_max = fx + stride[1] * output.dims()[3];
            col.slice_mut(s![.., .., fy, fx, .., ..])
                .assign(&input_.slice(s![.., .., fy..fy_max;stride[0], fx..fx_max;stride[1]]))
        }
    }

    let col = col.permuted_axes([1, 2, 3, 0, 4, 5]);
    let col = col
        .as_standard_layout()
        .into_shape([
            group,
            in_c_per_g * weight.dims()[2] * weight.dims()[3],
            input.dims()[0] * output.dims()[2] * output.dims()[3],
        ])
        .unwrap();

    for g in 0..group {
        linalg::general_mat_mul(
            1.0,
            &weight_.slice(s![g, .., ..]),
            &col.slice(s![g, .., ..]),
            1.0,
            &mut output_.slice_mut(s![g, .., ..]),
        );
    }

    let output_ = output_
        .into_shape([
            output.dims()[1],
            output.dims()[0],
            output.dims()[2],
            output.dims()[3],
        ])
        .unwrap()
        .permuted_axes([1, 0, 2, 3]);

    output.set_raw_vec(output_.as_standard_layout().to_owned().into_raw_vec());
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

    output.set_raw_vec(output_cuda.as_host_vec().unwrap());
}
