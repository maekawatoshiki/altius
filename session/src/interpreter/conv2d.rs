use altius_core::{
    node::{Conv2d, Node},
    tensor::Tensor,
};

#[cfg(feature = "cuda")]
use super::session::SafeCudnnContext;
use super::thread::ThreadCtx;
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
    pub inputs: &'a [&'a Tensor],
    pub outputs: &'a mut [Tensor],
    pub tctx: &'a ThreadCtx,
}

#[cfg(not(feature = "cuda"))]
pub fn compute(ctx: &mut Conv2dCtx) {
    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let output = &mut ctx.outputs[0];

    let kernel = &ctx.op.kernel_shape;
    let padding = &ctx.op.padding;
    let stride = &ctx.op.strides;
    let dilations = &ctx.op.dilations;

    let batch_size = input.dims()[0];
    let input_c = input.dims()[1];
    let input_h = input.dims()[2];
    let input_w = input.dims()[3];
    let input_hw = input_h * input_w;
    let output_h = output.dims()[2];
    let output_w = output.dims()[3];
    let output_hw = output_h * output_w;
    let _dilation = 1;
    let group = ctx.op.group as usize;
    let in_c_per_g = input_c / group;
    let out_c_per_g = output.dims()[1] / group;
    let stride_h = stride[0];
    let stride_w = stride[1];
    let dilation_h = dilations[0];
    let dilation_w = dilations[1];
    let kernel_h = kernel[0];
    let kernel_w = kernel[1];

    assert_eq!(dilations.len(), 2);
    assert!(padding.len() == 4);
    let pad_t = padding[0];
    let pad_l = padding[1];
    let _pad_b = padding[2];
    let _pad_r = padding[3];

    if let Some(bias) = ctx.inputs.get(Node::CONV2D_BIAS) {
        let mut output_ptr = output.data_mut::<f32>().as_mut_ptr();
        let bias_ptr_ = bias.data::<f32>().as_ptr();
        for _ in 0..batch_size {
            let mut bias_ptr = bias_ptr_;
            for _ in 0..group {
                for _ in 0..out_c_per_g {
                    for _ in 0..output_hw {
                        unsafe { *output_ptr = *bias_ptr };
                        output_ptr = unsafe { output_ptr.add(1) };
                    }
                    bias_ptr = unsafe { bias_ptr.add(1) };
                }
            }
        }
    } else {
        output.data_mut::<f32>().fill(0.);
    }

    let mut col = Tensor::uninit::<f32>(
        vec![
            batch_size, input_c, kernel[0], kernel[1], output_h, output_w,
        ]
        .into(),
    );

    assert!(input.dims().len() == 4);

    if pad_t == 0
        && pad_l == 0
        && stride_h == 1
        && stride_w == 1
        && dilation_h == 1
        && dilation_w == 1
    {
        let mut col_ptr = col.data_mut::<f32>().as_mut_ptr();
        let mut input_ptr = input.data::<f32>().as_ptr();

        let mut outer = batch_size * input_c;
        let inner = kernel_h * kernel_w;
        // Simple case
        while outer > 0 {
            let mut inner = inner;
            while inner > 0 {
                unsafe { std::ptr::copy_nonoverlapping(input_ptr, col_ptr, input_hw) }
                col_ptr = unsafe { col_ptr.add(output_hw) };
                inner -= 1
            }
            input_ptr = unsafe { input_ptr.add(input_hw) };
            outer -= 1
        }
    } else if dilation_h == 1 && dilation_w == 1 {
        let col = col.data_mut::<f32>();
        let input = input.data::<f32>();
        let _outer = batch_size * input_c;

        ctx.tctx.scope(|scope| {
            for (col, input) in col
                .chunks_mut(output_hw * kernel_h * kernel_w)
                .zip(input.chunks(input_hw))
            {
                scope.spawn(move || {
                    let input = input;
                    let mut col = col;

                    for fy in 0..kernel_h {
                        let ih = fy as isize - pad_t as isize;
                        for fx in 0..kernel_w {
                            let iw = fx as isize - pad_l as isize;
                            let mut ih = ih;
                            let mut owh = 0;
                            for _oh in 0..output_h {
                                if 0 <= ih && ih < input_h as isize {
                                    let jh = ih as usize * input_w;
                                    let mut iw = iw;
                                    for ow in 0..output_w {
                                        unsafe { *col.get_unchecked_mut(owh + ow) = 0. };
                                        if 0 <= iw && iw < input_w as isize {
                                            let jw = jh + iw as usize;
                                            unsafe {
                                                *col.get_unchecked_mut(owh + ow) =
                                                    *input.get_unchecked(jw)
                                            };
                                        }
                                        iw += stride_w as isize;
                                    }
                                } else {
                                    unsafe { col.get_unchecked_mut(owh..owh + output_w).fill(0.) };
                                }
                                owh += output_w;
                                ih += stride_h as isize;
                            }
                            col = unsafe { col.get_unchecked_mut(output_hw..) };
                        }
                    }
                });
            }
        })
    } else {
        let col = col.data_mut::<f32>();
        let input = input.data::<f32>();

        ctx.tctx.scope(|scope| {
            for (col, input) in col
                .chunks_mut(output_hw * kernel_h * kernel_w)
                .zip(input.chunks(input_hw))
            {
                scope.spawn(move || {
                    let input = input;
                    let mut col = col;

                    for fy in 0..kernel_h {
                        let ih = (fy * dilation_h) as isize - pad_t as isize;
                        for fx in 0..kernel_w {
                            let iw = (fx * dilation_w) as isize - pad_l as isize;
                            let mut ih = ih;
                            let mut owh = 0;
                            for _oh in 0..output_h {
                                if 0 <= ih && ih < input_h as isize {
                                    let jh = ih as usize * input_w;
                                    let mut iw = iw;
                                    for ow in 0..output_w {
                                        unsafe { *col.get_unchecked_mut(owh + ow) = 0. };
                                        if 0 <= iw && iw < input_w as isize {
                                            let jw = jh + iw as usize;
                                            unsafe {
                                                *col.get_unchecked_mut(owh + ow) =
                                                    *input.get_unchecked(jw)
                                            };
                                        }
                                        iw += stride_w as isize;
                                    }
                                } else {
                                    unsafe { col.get_unchecked_mut(owh..owh + output_w).fill(0.) };
                                }
                                owh += output_w;
                                ih += stride_h as isize;
                            }
                            col = unsafe { col.get_unchecked_mut(output_hw..) };
                        }
                    }
                });
            }
        });
    }

    let mut col_ptr = col.data::<f32>().as_ptr();
    let col_stride = in_c_per_g * kernel[0] * kernel[1] * output_h * output_w;
    let weight_ptr = weight.data::<f32>().as_ptr();
    let weight_stride = out_c_per_g * in_c_per_g * kernel[0] * kernel[1];
    let mut output_ptr = output.data_mut::<f32>().as_mut_ptr();
    let output_stride = out_c_per_g * output_hw;
    let k = in_c_per_g * kernel[0] * kernel[1];

    for _ in 0..batch_size {
        let mut weight_ptr = weight_ptr;
        for _ in 0..group {
            unsafe {
                #[cfg(not(feature = "cblas"))]
                matrixmultiply::sgemm(
                    out_c_per_g,
                    k,
                    output_hw,
                    1.0,
                    weight_ptr,
                    k as isize,
                    1,
                    col_ptr,
                    output_hw as isize,
                    1,
                    1.0,
                    output_ptr,
                    output_hw as isize,
                    1,
                );
                #[cfg(feature = "cblas")]
                {
                    cblas_sys::cblas_sgemm(
                        cblas_sys::CblasRowMajor,
                        cblas_sys::CblasNoTrans,
                        cblas_sys::CblasNoTrans,
                        out_c_per_g as i32,
                        output_hw as i32,
                        k as i32,
                        1.0f32,
                        weight_ptr,
                        k as i32,
                        col_ptr as *const _,
                        output_hw as i32,
                        1.0f32,
                        output_ptr as *mut f32,
                        output_hw as i32,
                    );
                }
                col_ptr = col_ptr.add(col_stride);
                weight_ptr = weight_ptr.add(weight_stride);
                output_ptr = output_ptr.add(output_stride);
            };
        }
    }
}

#[cfg(feature = "cuda")]
pub fn compute(ctx: &mut Conv2dCtx) {
    let conv = ctx.op;
    let input = &ctx.inputs[Node::CONV2D_IN];
    let weight = &ctx.inputs[Node::CONV2D_WEIGHT];
    let bias = ctx.inputs.get(Node::CONV2D_BIAS).map_or(
        Tensor::zeros::<f32>(vec![weight.dims()[0]].into()),
        |&bias| bias.clone(),
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
    let mut output_cuda =
        unsafe { DeviceBuffer::uninitialized(output.dims().total_elems()).unwrap() };
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
