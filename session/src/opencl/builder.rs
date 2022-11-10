use std::ptr;

use altius_core::{
    model::Model,
    node::{compute_output_shapes, Op},
    tensor::{Tensor, TypedShape},
};
use opencl3::{
    command_queue::{enqueue_read_buffer, CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{self, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::CL_BLOCKING,
};
use rustc_hash::FxHashMap;

use crate::SessionError;

use super::{
    session::{ExecutionPlan, OpenclSession},
    tensor::OpenclTensor,
};

#[derive(Default)]
pub struct OpenclSessionBuilder<'a> {
    #[allow(dead_code)] // TODO: Remove later.
    model: Option<&'a Model>,
}

impl<'a> OpenclSessionBuilder<'a> {
    pub const fn new() -> Self {
        Self { model: None }
    }

    pub fn with_model(mut self, model: &'a Model) -> Self {
        self.model = Some(model);
        self
    }

    pub fn build(self) -> Result<OpenclSession<'a>, SessionError> {
        let model = self
            .model
            .ok_or(SessionError::Message("Model is not set".to_string()))?;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| SessionError::Message(format!("Failed to get device: {}", e.to_string())))?
            .first()
            .ok_or_else(|| SessionError::Message("No device found".into()))?;
        let device = Device::new(device_id);

        let context = Context::from_device(&device).map_err(|e| {
            SessionError::Message(format!("Failed to create context: {}", e.to_string()))
        })?;

        let queue = unsafe {
            CommandQueue::create_with_properties(
                &context,
                device_id,
                CL_QUEUE_PROFILING_ENABLE,
                0, /* TODO: What does queue_size mean? */
            )
        }
        .map_err(|e| {
            SessionError::Message(format!("Failed to create command queue: {}", e.to_string()))
        })?;

        let sorted_nodes = model.topo_sort_nodes();
        let mut values = FxHashMap::default();

        // Allocate initializers.
        for (init_id, tensor) in model.inits.clone() {
            let buf = unsafe {
                memory::create_buffer(
                    context.get(),
                    CL_MEM_READ_ONLY,
                    tensor.elem_ty().size() * tensor.dims().total_elems(),
                    ptr::null_mut(),
                )
            }
            .map_err(|e| SessionError::Message(format!("Failed to create buffer: {e:?}")))?;

            let _event = unsafe {
                enqueue_read_buffer(
                    queue.get(),
                    buf,
                    CL_BLOCKING,
                    0,
                    tensor.elem_ty().size() * tensor.dims().total_elems(),
                    tensor.data_as_ptr() as *mut _,
                    0,
                    ptr::null(),
                )
                .unwrap()
            };

            values.insert(init_id, OpenclTensor { tensor, buf });
        }

        // Allocate inputs.
        for &input_id in &model.inputs {
            if model.inits.contains_key(&input_id) {
                continue;
            }

            let input = &model.values.inner()[input_id];
            let Some(shape) = input.shape.as_ref() else {
                return Err(SessionError::Message(format!("Unknown shape input")));
            };

            let buf = unsafe {
                memory::create_buffer(
                    context.get(),
                    CL_MEM_READ_ONLY,
                    shape.elem_ty.size() * shape.dims.total_elems(),
                    ptr::null_mut(),
                )
            }
            .map_err(|e| SessionError::Message(format!("Failed to create buffer: {e:?}")))?;
            let tensor = Tensor::empty_of_type(shape.elem_ty, shape.dims.clone());
            values.insert(input_id, OpenclTensor { tensor, buf });
        }

        let mut execution_plans = vec![];

        for node_id in sorted_nodes {
            let node = &model.nodes[node_id];
            let mut op = node.op.clone();

            let inputs = node
                .inputs
                .iter()
                .map(|input| values.get(input).unwrap())
                .collect::<Vec<_>>();
            let output_shapes = compute_output_shapes(
                &mut op,
                inputs
                    .iter()
                    .map(|t| &t.tensor)
                    .collect::<Vec<_>>()
                    .as_slice(),
                model.opset_version,
            );

            for (TypedShape { elem_ty, dims }, &output_id) in
                output_shapes.into_iter().zip(node.outputs.iter())
            {
                let buf = unsafe {
                    memory::create_buffer(
                        context.get(),
                        CL_MEM_READ_WRITE, // TODO: Is this appropriate?
                        elem_ty.size() * dims.total_elems(),
                        ptr::null_mut(),
                    )
                }
                .map_err(|e| {
                    SessionError::Message(format!("Failed to create buffer: {}", e.to_string()))
                })?;
                values.insert(
                    output_id,
                    OpenclTensor {
                        tensor: Tensor::empty_of_type(elem_ty, dims),
                        buf,
                    },
                );
            }

            // TODO: Actual compilation performs here...
            let kernel = match &node.op {
                Op::Add => compile_add(&context)?,
                op => todo!("{op:?}"),
            };
            execution_plans.push(ExecutionPlan { kernel, node_id });
        }

        Ok(OpenclSession {
            model,
            device,
            context,
            queue,
            values,
            execution_plans,
        })
    }
}

fn compile_add(context: &Context) -> Result<Kernel, SessionError> {
    let name = "add";
    let program = r#"
kernel void add(global float *out,
                global float const *in_0,
                global float const *in_1) {
    const size_t i = get_global_id(0);
    out[i] = in_0[i] + in_1[i];
}"#;
    let program = Program::create_and_build_from_source(context, program, "").map_err(|e| {
        SessionError::Message(format!("Failed to compile kernel: {}", e.to_string()))
    })?;
    let kernel = Kernel::create(&program, name).map_err(|e| {
        SessionError::Message(format!("Failed to create kernel: {}", e.to_string()))
    })?;

    Ok(kernel)
}

#[cfg(test)]
fn is_opencl_supported() -> bool {
    let Ok(devices) = get_all_devices(CL_DEVICE_TYPE_GPU) else { return false };
    !devices.is_empty()
}

#[test]
fn test_build() {
    if !is_opencl_supported() {
        return;
    }

    let model = Model::default();
    let _ = OpenclSessionBuilder::new().with_model(&model).build();
}

#[test]
fn test_build_add() {
    use crate::interpreter::InterpreterSession;
    use altius_core::{
        node::{Node, Op},
        tensor::{TensorElemType, TypedShape},
    };
    use std::time::Instant;

    if !is_opencl_supported() {
        return;
    }

    let mut model = Model::default();

    let n = 8;
    let in_0 = model.values.new_val_named_and_shaped(
        "x".to_owned(),
        TypedShape::new(vec![n, 1024, 8, 8].into(), TensorElemType::F32),
    );
    let in_1 = model.values.new_val_named_and_shaped(
        "y".to_owned(),
        TypedShape::new(vec![n, 1024, 8, 8].into(), TensorElemType::F32),
    );
    let out = model.values.new_val_named_and_shaped(
        "z".to_owned(),
        TypedShape::new(vec![n, 1024, 8, 8].into(), TensorElemType::F32),
    );
    model.add_node(Node::new(Op::Add).with_ins(vec![in_0, in_1]).with_out(out));

    model.inputs.push(in_0);
    model.inputs.push(in_1);
    model.outputs.push(out);

    let opencl_sess = OpenclSessionBuilder::new()
        .with_model(&model)
        .build()
        .unwrap();
    let cpu_sess = InterpreterSession::new(&model);

    let x = Tensor::rand::<f32>(vec![n, 1024, 8, 8].into());
    let y = Tensor::rand::<f32>(vec![n, 1024, 8, 8].into());
    let opencl_z = opencl_sess
        .run(vec![(in_0, x.clone()), (in_1, y.clone())])
        .unwrap();
    let start = Instant::now();
    let cpu_z = cpu_sess.run(vec![(in_0, x), (in_1, y)]).unwrap();
    println!("cpu {:?}", start.elapsed());

    assert!(allclose(opencl_z[0].data::<f32>(), cpu_z[0].data::<f32>()));
}

#[cfg(test)]
fn allclose(x: &[f32], y: &[f32]) -> bool {
    let atol = 1e-5;
    let rtol = 1e-5;

    if x.len() != y.len() {
        return false;
    }

    x.iter().zip(y.iter()).all(|(x, y)| {
        ((x - y).abs() <= (atol + rtol * y.abs()))
            || (x.is_infinite() && y.is_infinite() && x.is_sign_positive() == y.is_sign_positive())
    })
}
