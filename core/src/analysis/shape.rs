use std::borrow::Cow;

use rustc_hash::FxHashMap;
use thiserror::Error;

use crate::{
    fixed_dim::FixedDimensions,
    model::Model,
    node::NodeId,
    op::Op,
    tensor::{Tensor, TensorElemType, TypedFixedShape},
    value::ValueId,
};

#[derive(Debug, Clone, Error)]
pub enum ShapeError {
    #[error("Something went wrong: {0}")]
    Message(Cow<'static, str>),
}

impl Op {
    /// Computes the output shapes of the operation.
    /// `self` could be overwritten. (e.g. if auto_pad is given, conv paddings are modified)
    pub fn compute_output_shapes(
        &mut self,
        inputs: &[&Tensor],
        num_outputs: usize,
        opset_version: i64,
    ) -> Result<Vec<TypedFixedShape>, ShapeError> {
        let mut shapes = vec![];

        match self {
            Op::Conv2d(conv) => {
                let auto_pad = &conv.auto_pad;
                let kernel = &conv.kernel_shape;
                let stride = &conv.strides;
                let padding = &conv.padding;
                let dilations = &conv.dilations;
                let input = inputs[Op::CONV2D_IN].dims();
                let weight = inputs[Op::CONV2D_WEIGHT].dims();

                assert_eq!(dilations.len(), 2);

                let pad_h;
                let pad_w;
                if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                    let out0 = (input[2] as f32 / stride[0] as f32).ceil() as usize;
                    let out1 = (input[3] as f32 / stride[1] as f32).ceil() as usize;
                    let pad0 =
                        ((out0 - 1) * stride[0] + ((kernel[0] - 1) + 1)).saturating_sub(input[2]);
                    let pad1 =
                        ((out1 - 1) * stride[1] + ((kernel[1] - 1) + 1)).saturating_sub(input[3]);
                    assert!(auto_pad == "SAME_UPPER");
                    let new_padding = vec![pad0 / 2, pad1 / 2, pad0 - pad0 / 2, pad1 - pad1 / 2];
                    conv.padding = new_padding.into();
                    pad_h = pad0;
                    pad_w = pad1;
                } else if padding.len() == 2 {
                    pad_h = padding[0] * 2;
                    pad_w = padding[1] * 2;
                    conv.padding = vec![padding[0], padding[1], padding[0], padding[1]].into();
                } else if padding.len() == 4 {
                    pad_h = padding[0] + padding[2];
                    pad_w = padding[1] + padding[3];
                } else {
                    return Err(ShapeError::Message(
                        format!("Conv2d: Unknown padding pattern: {padding:?}").into(),
                    ));
                }

                let h_in = input[2];
                let w_in = input[3];

                let output_shape = vec![
                    input[0],
                    weight[0],
                    (h_in + pad_h - dilations[0] * (kernel[0] - 1) - 1) / stride[0] + 1,
                    (w_in + pad_w - dilations[1] * (kernel[1] - 1) - 1) / stride[1] + 1,
                ];
                shapes.push(TypedFixedShape::new(
                    output_shape.into(),
                    inputs[Op::CONV2D_IN].elem_ty(),
                ));
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow => {
                let x = inputs[0].dims();
                let y = inputs[1].dims();
                let shape = x.broadcast(y).unwrap();
                shapes.push(TypedFixedShape::new(shape, inputs[0].elem_ty()));
            }
            Op::Greater => {
                let x = inputs[0].dims();
                let y = inputs[1].dims();
                let shape = x.broadcast(y).unwrap();
                shapes.push(TypedFixedShape::new(shape, TensorElemType::Bool));
            }
            Op::Where => {
                let x = inputs[1].dims();
                let y = inputs[2].dims();
                let shape = x.broadcast(y).unwrap();
                shapes.push(TypedFixedShape::new(shape, inputs[1].elem_ty()));
            }
            Op::MaxPool(maxpool) => {
                let auto_pad = &maxpool.auto_pad;
                let kernel = &maxpool.kernel_shape;
                let stride = &maxpool.strides;
                let input = &inputs[Op::MAXPOOL_IN].dims();
                let mut padding = &maxpool.padding;

                if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                    let out0 = (input[2] as f32 / stride[0] as f32).ceil() as usize;
                    let out1 = (input[3] as f32 / stride[1] as f32).ceil() as usize;
                    let pad0 =
                        ((out0 - 1) * stride[0] + ((kernel[0] - 1) + 1)).saturating_sub(input[2]);
                    let pad1 =
                        ((out1 - 1) * stride[1] + ((kernel[1] - 1) + 1)).saturating_sub(input[3]);
                    assert!(auto_pad == "SAME_UPPER");
                    let new_padding = vec![pad0 / 2, pad1 / 2, pad0 - pad0 / 2, pad1 - pad1 / 2];
                    maxpool.padding = new_padding.into();
                    padding = &maxpool.padding;
                }

                let h_in = input[2];
                let w_in = input[3];
                let output_shape = vec![
                    input[0],
                    input[1],
                    (h_in + (padding[0] + padding[2]) - (kernel[0] - 1) - 1) / stride[0] + 1,
                    (w_in + (padding[1] + padding[3]) - (kernel[1] - 1) - 1) / stride[1] + 1,
                ];
                shapes.push(TypedFixedShape::new(
                    output_shape.into(),
                    inputs[Op::MAXPOOL_IN].elem_ty(),
                ));
            }
            Op::GlobalAveragePool => {
                let input = &inputs[Op::GLOBALAVERAGEPOOL_IN].dims();
                assert!(input.len() == 4);
                shapes.push(TypedFixedShape::new(
                    vec![input[0], input[1], 1, 1].into(),
                    inputs[Op::GLOBALAVERAGEPOOL_IN].elem_ty(),
                ));
            }
            Op::Expand => {
                let input = inputs[0];
                let shape = inputs[1];
                shapes.push(TypedFixedShape::new(
                    shape
                        .data::<i64>()
                        .iter()
                        .map(|&x| x as usize)
                        .collect::<Vec<_>>()
                        .into(),
                    input.elem_ty(),
                ));
            }
            Op::Range => {
                if !inputs[0].elem_ty().is_i64() {
                    return Err(ShapeError::Message(
                        "Range only supports i64 for now".into(),
                    ));
                }

                let start = inputs[0].data::<i64>()[0];
                let limit = inputs[1].data::<i64>()[0];
                let delta = inputs[2].data::<i64>()[0];
                let num_elems = (((limit - start) as f32 / delta as f32).ceil() as i64).max(0);
                shapes.push(TypedFixedShape::new(
                    vec![num_elems as usize].into(),
                    inputs[0].elem_ty(),
                ));
            }
            Op::Reshape => {
                let shape = inputs[Op::RESHAPE_SHAPE]
                    .data::<i64>()
                    .iter()
                    .map(|&x| {
                        if x == -1 {
                            let other_dims_sz: i64 = inputs[Op::RESHAPE_SHAPE]
                                .data::<i64>()
                                .iter()
                                .filter(|x| **x != -1)
                                .product();
                            inputs[Op::RESHAPE_IN].dims().total_elems() / other_dims_sz as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect::<Vec<_>>();
                shapes.push(TypedFixedShape::new(
                    shape.into(),
                    inputs[Op::RESHAPE_IN].elem_ty(),
                ))
            }
            Op::Flatten(flatten) => {
                let dims = inputs[Op::FLATTEN_IN].dims();
                assert!(flatten.axis >= 0);
                let x: FixedDimensions = dims[..flatten.axis as usize].to_vec().into();
                let y: FixedDimensions = dims[flatten.axis as usize..].to_vec().into();
                shapes.push(TypedFixedShape::new(
                    vec![x.total_elems(), y.total_elems()].into(),
                    inputs[Op::FLATTEN_IN].elem_ty(),
                ));
            }
            Op::Resize(resize) => {
                if inputs.len() == 4 {
                    let sizes = &inputs[Op::RESIZE_IN_SIZES];
                    assert!(sizes.dims().len() == 1 && sizes.dims()[0] == 4);
                    shapes.push(TypedFixedShape::new(
                        FixedDimensions::from_i64(sizes.data::<i64>()),
                        inputs[Op::RESIZE_IN_X].elem_ty(),
                    ))
                } else if inputs.len() == 3 {
                    // TODO: Support other cases.
                    assert!(resize.coordinate_transformation_mode == "asymmetric");
                    assert!(resize.mode == "nearest");
                    assert!(resize.nearest_mode == "floor");
                    assert!(inputs.len() == 3);
                    let x = &inputs[Op::RESIZE_IN_X];
                    assert!(x.dims().len() == 4);
                    let _roi = &inputs[Op::RESIZE_IN_ROI].data::<f32>(); // TODO: According to https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize,
                                                                         // it only takes effect when coordinate_transformation_mode is "tf_crop_and_resize".
                                                                         // Since we assume coordinate_transformation_mode is "asymmetric" for now, just ignore roi.
                    let scales = &inputs[Op::RESIZE_IN_SCALES].data::<f32>();
                    shapes.push(TypedFixedShape::new(
                        vec![
                            (x.dims()[0] as f32 * scales[0]).floor() as usize,
                            (x.dims()[1] as f32 * scales[1]).floor() as usize,
                            (x.dims()[2] as f32 * scales[2]).floor() as usize,
                            (x.dims()[3] as f32 * scales[3]).floor() as usize,
                        ]
                        .into(),
                        inputs[Op::RESIZE_IN_X].elem_ty(),
                    ))
                    // NOTE: Use the following code when roi takes effect ... right?
                    // shapes.push(TypedFixedShape::new(
                    //     vec![
                    //         (x.dims()[0] as f32 * (roi[4] - roi[0]) * scales[0]).floor() as usize,
                    //         (x.dims()[1] as f32 * (roi[5] - roi[1]) * scales[1]).floor() as usize,
                    //         (x.dims()[2] as f32 * (roi[6] - roi[2]) * scales[2]).floor() as usize,
                    //         (x.dims()[3] as f32 * (roi[7] - roi[3]) * scales[3]).floor() as usize,
                    //     ]
                    //     .into(),
                    //     inputs[Op::RESIZE_IN_X].elem_ty(),
                    // ))
                } else {
                    return Err(ShapeError::Message("Resize: Unsupported pattern".into()));
                }
            }
            Op::Concat(concat) => {
                let mut dims = inputs[Op::CONCAT_IN].dims().clone();
                let mut sum = 0;
                for i in inputs {
                    sum += i.dims()[concat.axis as usize];
                }
                dims.as_mut_slice()[concat.axis as usize] = sum;
                shapes.push(TypedFixedShape::new(dims, inputs[Op::CONCAT_IN].elem_ty()))
            }
            Op::Transpose(trans) => {
                assert!(!trans.perm.is_empty());
                let in_dims = inputs[Op::TRANSPOSE_IN].dims().as_slice();
                let mut dims = vec![0usize; in_dims.len()];
                for i in 0..in_dims.len() {
                    dims[i] = in_dims[trans.perm[i] as usize];
                }
                shapes.push(TypedFixedShape::new(
                    dims.into(),
                    inputs[Op::TRANSPOSE_IN].elem_ty(),
                ))
            }
            Op::Squeeze(squeeze) => {
                if opset_version < 12 {
                    assert!(!squeeze.axes.is_empty());
                    assert!(squeeze.axes.iter().all(|&x| x >= 0));
                    let in_dims = inputs[Op::SQUEEZE_IN].dims().as_slice();
                    let mut dims = vec![];
                    for (i, &x) in in_dims.iter().enumerate() {
                        if squeeze.axes.contains(&(i as i64)) {
                            continue;
                        }
                        dims.push(x);
                    }
                    shapes.push(TypedFixedShape::new(
                        dims.into(),
                        inputs[Op::SQUEEZE_IN].elem_ty(),
                    ))
                } else {
                    let in_dims = inputs[0].dims().as_slice();
                    let axes = inputs[1].data::<i64>();
                    assert!(!axes.is_empty());
                    assert!(axes.iter().all(|&x| x >= 0));
                    let mut dims = vec![];
                    for (i, &x) in in_dims.iter().enumerate() {
                        if axes.contains(&(i as i64)) {
                            continue;
                        }
                        dims.push(x);
                    }
                    shapes.push(TypedFixedShape::new(dims.into(), inputs[0].elem_ty()))
                }
            }
            Op::Unsqueeze(unsqueeze) => {
                if opset_version >= 13 {
                    // axes is no longer an attribute but node input.
                    let in_dims = inputs[0].dims().as_slice().to_vec();
                    let axes = inputs[1].data::<i64>();
                    let mut dims = in_dims;
                    for &x in axes {
                        dims.insert(x as usize, 1);
                    }
                    shapes.push(TypedFixedShape::new(dims.into(), inputs[0].elem_ty()))
                } else {
                    let in_dims = inputs[Op::UNSQUEEZE_IN].dims().as_slice().to_vec();
                    let mut dims = in_dims;
                    for &x in unsqueeze.axes.iter() {
                        dims.insert(x as usize, 1);
                    }
                    shapes.push(TypedFixedShape::new(
                        dims.into(),
                        inputs[Op::UNSQUEEZE_IN].elem_ty(),
                    ))
                }
            }
            Op::ReduceMin(rmin) => {
                let in_dims = inputs[0].dims();
                let keepdims = if rmin.keep_dims { Some(1) } else { None };
                let axes = rmin
                    .axes
                    .iter()
                    .map(|&axis| {
                        if axis < 0 {
                            (in_dims.len() as i64 + axis) as usize
                        } else {
                            axis as usize
                        }
                    })
                    .collect::<Vec<_>>();
                let mut dims = in_dims
                    .as_slice()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| if axes.contains(&i) { keepdims } else { Some(d) })
                    .collect::<Vec<_>>();
                if dims.is_empty() {
                    dims.push(1);
                }
                shapes.push(TypedFixedShape::new(dims.into(), inputs[0].elem_ty()))
            }
            Op::ReduceMax(rmax) => {
                assert!(opset_version <= 13);
                let in_dims = inputs[0].dims();
                let keepdims = if rmax.keep_dims { Some(1) } else { None };
                let axes = if rmax.axes.is_empty() {
                    in_dims
                        .iter()
                        .enumerate()
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                } else {
                    rmax.axes
                        .iter()
                        .map(|&axis| {
                            if axis < 0 {
                                (in_dims.len() as i64 + axis) as usize
                            } else {
                                axis as usize
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let mut dims = in_dims
                    .as_slice()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| if axes.contains(&i) { keepdims } else { Some(d) })
                    .collect::<Vec<_>>();
                if dims.is_empty() {
                    dims.push(1);
                }
                shapes.push(TypedFixedShape::new(dims.into(), inputs[0].elem_ty()))
            }
            Op::ReduceMean(rmean) => {
                let in_dims = inputs[0].dims();
                let keepdims = if rmean.keep_dims { Some(1) } else { None };
                let axes = rmean
                    .axes
                    .iter()
                    .map(|&axis| {
                        if axis < 0 {
                            (in_dims.len() as i64 + axis) as usize
                        } else {
                            axis as usize
                        }
                    })
                    .collect::<Vec<_>>();
                let mut dims = in_dims
                    .as_slice()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| if axes.contains(&i) { keepdims } else { Some(d) })
                    .collect::<Vec<_>>();
                if dims.is_empty() {
                    dims.push(1);
                }
                shapes.push(TypedFixedShape::new(dims.into(), inputs[0].elem_ty()))
            }
            Op::Loop => {
                assert!(inputs.len() == 3);
                let _m = inputs[0].data::<i64>();
                let cond = inputs[1].data::<u8>();
                assert!(cond[0] == 1);
                let v_initial = inputs[2].data::<i32>();
                assert!(v_initial[0] == 0);
                return Err(ShapeError::Message("Loop: Unsupported op".into()));
                // shapes.push(vec![1].into());
                // shapes.push(vec![m[0] as usize].into());
            }
            Op::Tile => {
                let in_dims = inputs[Op::TILE_IN].dims();
                let reps = inputs[Op::TILE_REPEATS].data::<i64>();
                let mut dims = vec![];
                for (i, &x) in in_dims.as_slice().iter().enumerate() {
                    dims.push(x * reps[i] as usize);
                }
                shapes.push(TypedFixedShape::new(
                    dims.into(),
                    inputs[Op::TILE_IN].elem_ty(),
                ));
            }
            Op::Split(split) => {
                if opset_version >= 13 {
                    let axis = split.axis;
                    assert!(axis >= 0, "Negative index not supported");
                    let input = inputs[0].dims();
                    let split = inputs[1].data::<i64>();
                    for s in split {
                        let mut dims = input.clone();
                        dims[axis as usize] = *s as usize;
                        shapes.push(TypedFixedShape::new(dims, inputs[0].elem_ty()));
                    }
                } else {
                    let axis = split.axis;
                    let split = &split.split;
                    assert!(axis >= 0, "Negative index not supported");
                    let input = inputs[0].dims();
                    for s in split {
                        let mut dims = input.clone();
                        dims[axis as usize] = *s as usize;
                        shapes.push(TypedFixedShape::new(dims, inputs[0].elem_ty()));
                    }
                }
            }
            Op::Slice => {
                let in_data_dims = inputs[Op::SLICE_IN_DATA].dims();
                let in_starts = inputs[Op::SLICE_IN_STARTS].data::<i64>();
                let in_ends = inputs[Op::SLICE_IN_ENDS].data::<i64>();
                let in_axes = inputs[Op::SLICE_IN_AXES].data::<i64>();
                let ones = vec![1i64; in_axes.len()];
                let in_steps = inputs
                    .get(Op::SLICE_IN_STEPS)
                    .map_or(ones.as_slice(), |s| s.data::<i64>());
                assert!(in_starts.iter().all(|&x| x >= 0));
                assert!(in_ends.iter().all(|&x| x >= 0));
                assert!(in_steps.iter().all(|&x| x >= 0));
                let mut dims = in_data_dims.clone();
                for (((start, end), &axis), step) in in_starts
                    .iter()
                    .zip(in_ends.iter())
                    .zip(in_axes.iter())
                    .zip(in_steps.iter())
                {
                    let start = *start as usize;
                    let end = *end as usize;
                    let axis = if axis < 0 {
                        (in_data_dims.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let step = *step as usize;
                    let out_dim = (end - start) / step;
                    assert!(out_dim > 0);
                    dims[axis] = out_dim;
                }
                shapes.push(TypedFixedShape::new(
                    dims,
                    inputs[Op::SLICE_IN_DATA].elem_ty(),
                ))
            }
            Op::Gather(gather) => {
                let mut data = inputs[0].dims().0.to_owned();
                let indices = inputs[1].dims();
                assert!(gather.axis >= 0);
                assert!(
                    indices.is_scalar() || (indices.len() == 2 && indices[0] == 1),
                    "Unsupported indices shape: {indices:?}"
                );
                if indices.is_scalar() {
                    data.remove(gather.axis as usize);
                    shapes.push(TypedFixedShape::new(data.into(), inputs[0].elem_ty()))
                } else {
                    assert_eq!(gather.axis, 0);
                    data.remove(gather.axis as usize);
                    data.insert(0, 1);
                    data.insert(1, indices[1]);
                    shapes.push(TypedFixedShape::new(data.into(), inputs[0].elem_ty()))
                }
            }
            Op::Shape(shape) => {
                assert!(shape.end.is_none());
                assert!(shape.start == 0);
                let input = inputs[0].dims();
                shapes.push(TypedFixedShape::new(
                    vec![input.len()].into(),
                    TensorElemType::I64,
                ));
            }
            Op::NonMaxSuppression => {
                return Err(ShapeError::Message(
                    "NonMaxSuppression: Unsupported op".into(),
                ))
            }
            Op::MatMul => {
                let in_a = &inputs[Op::MATMUL_IN_A].dims();
                let in_b = &inputs[Op::MATMUL_IN_B].dims();
                assert!(
                    in_a[1] == in_b[0]
                        || (in_a.len() == 3 && in_b.len() == 2 && in_a[2] == in_b[0])
                        || (in_a.len() == 3
                            && in_b.len() == 3
                            && in_a[0] == in_b[0]
                            && in_a[2] == in_b[1])
                        || (in_a.len() == 4
                            && in_b.len() == 4
                            && in_a[0] == 1
                            && in_b[0] == 1
                            && in_a[1] == in_b[1]),
                    "A shape: {in_a:?}, B shape: {in_b:?}"
                );
                if in_a.len() == 4 && in_b.len() == 4 {
                    shapes.push(TypedFixedShape::new(
                        vec![in_a[0], in_a[1], in_a[2], in_b[3]].into(),
                        inputs[Op::MATMUL_IN_A].elem_ty(),
                    ));
                } else if in_a.len() == 3 && in_b.len() == 2 {
                    shapes.push(TypedFixedShape::new(
                        vec![in_a[0], in_a[1], in_b[1]].into(),
                        inputs[Op::MATMUL_IN_A].elem_ty(),
                    ));
                } else if in_a.len() == 3 && in_b.len() == 3 {
                    shapes.push(TypedFixedShape::new(
                        vec![in_a[0], in_a[1], in_b[2]].into(),
                        inputs[Op::MATMUL_IN_A].elem_ty(),
                    ));
                } else {
                    shapes.push(TypedFixedShape::new(
                        vec![in_a[0], in_b[1]].into(),
                        inputs[Op::MATMUL_IN_A].elem_ty(),
                    ));
                }
            }
            Op::Gemm(gemm) => {
                let in_a = &inputs[Op::GEMM_IN_A].dims();
                let (in_a0, in_a1) = if gemm.trans_a {
                    (in_a[1], in_a[0])
                } else {
                    (in_a[0], in_a[1])
                };
                let in_b = &inputs[Op::GEMM_IN_B].dims();
                let (in_b0, in_b1) = if gemm.trans_b {
                    (in_b[1], in_b[0])
                } else {
                    (in_b[0], in_b[1])
                };
                assert_eq!(in_a1, in_b0);
                shapes.push(TypedFixedShape::new(
                    vec![in_a0, in_b1].into(),
                    inputs[Op::GEMM_IN_A].elem_ty(),
                ));
            }
            Op::Constant(_) => return Err(ShapeError::Message("Constant: Unsupported op".into())),
            // Element-wise operations.
            Op::Sqrt
            | Op::ReLU
            | Op::LeakyReLU(_)
            | Op::Gelu
            | Op::Sigmoid
            | Op::Erf
            | Op::Tanh
            | Op::Clip
            | Op::HardSigmoid(_)
            | Op::Round
            | Op::Exp
            | Op::Softmax(_)
            | Op::BatchNormalization(_) => {
                let input = inputs[0];
                shapes.push(TypedFixedShape::new(input.dims().clone(), input.elem_ty()));
            }
            Op::LayerNormalization(ln) => {
                assert!(num_outputs == 1);
                let input = inputs[0];
                assert!(ln.stash_type == 1);
                shapes.push(TypedFixedShape::new(input.dims().clone(), input.elem_ty()));
            }
            Op::Cast(cast) => {
                let input = inputs[0];
                shapes.push(TypedFixedShape::new(input.dims().clone(), cast.to));
            }
            Op::FusedElemwise(ref mut f) => {
                let mut map = FxHashMap::default();
                for (i, val_id) in f.input_map.iter().enumerate() {
                    map.insert(*val_id, inputs[i]);
                }
                let mut prev_output_shape = vec![];
                let mut prev_output_id = None;
                for (i, (op, inputs, output)) in f.chain.iter_mut().enumerate() {
                    assert_eq!(output.len(), 1);
                    if i == 0 {
                        let ins = inputs.iter().map(|v| map[v]).collect::<Vec<_>>();
                        prev_output_shape =
                            op.compute_output_shapes(&ins, output.len(), opset_version)?;
                    } else {
                        let prev_output_dummy = Tensor::empty_of_type(
                            prev_output_shape[0].elem_ty,
                            prev_output_shape[0].dims.clone(),
                        );
                        prev_output_shape = op.compute_output_shapes(
                            &inputs
                                .iter()
                                .map(|v| {
                                    map.get(v).copied().unwrap_or_else(|| {
                                        assert_eq!(Some(*v), prev_output_id);
                                        &prev_output_dummy
                                    })
                                })
                                .collect::<Vec<_>>(),
                            output.len(),
                            opset_version,
                        )?;
                    }
                    prev_output_id = Some(output[0]);
                }
                shapes.extend(prev_output_shape.into_iter());
            }
        }

        Ok(shapes)
    }
}

/// Infer `TypedFixedShape`s of output tensors for each node.
/// It skips to infer on nodes without information for inference.
pub fn infer_shapes(
    model: &Model,
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedFixedShape>)>,
    value_shapes: &mut FxHashMap<ValueId, TypedFixedShape>,
) -> Result<(), ShapeError> {
    let sorted_nodes = model.topo_sort_nodes();
    let mut values = model.inits.clone();

    for &val_id in &model.inputs {
        let shape = &model.values.inner()[val_id].shape;
        let Some(shape) = shape else { continue };
        let tensor = Tensor::zeros_of_type(
            shape.elem_ty,
            shape
                .dims
                .as_fixed_dims()
                .ok_or_else(|| ShapeError::Message("Must be fixed dimension".into()))?,
        );
        values.insert(val_id, tensor);
    }

    for node in sorted_nodes {
        infer_shape(model, &mut values, shapes, node)?
    }

    value_shapes.extend(values.into_iter().map(|(val_id, tensor)| {
        let dims = tensor.dims().clone();
        let elem_ty = tensor.elem_ty();
        let shape = TypedFixedShape::new(dims, elem_ty);
        (val_id, shape)
    }));

    Ok(())
}

fn infer_shape(
    model: &Model,
    values: &mut FxHashMap<ValueId, Tensor>,
    shapes: &mut FxHashMap<NodeId, (Op, Vec<TypedFixedShape>)>,
    node_id: NodeId,
) -> Result<(), ShapeError> {
    let node = &model.nodes[node_id];
    let mut op = node.op.clone();
    let mut inputs = vec![];
    for input in &node.inputs {
        let Some(input) = values.get(input) else { return Ok(()); };
        inputs.push(input);
    }
    let output_shapes =
        op.compute_output_shapes(&inputs, node.outputs.len(), model.opset_version)?;
    let mut outputs = vec![];
    for shape in &output_shapes {
        outputs.push(Tensor::empty_of_type(shape.elem_ty, shape.dims.clone()));
    }
    for (&val, output) in node.outputs.iter().zip(outputs.into_iter()) {
        values.insert(val, output);
    }
    shapes.insert(node_id, (op, output_shapes));
    Ok(())
}
