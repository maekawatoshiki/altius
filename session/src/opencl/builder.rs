use std::ptr;

use altius_core::{
    model::Model,
    node::{compute_output_shapes, Op},
    tensor::{Tensor, TypedShape},
};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{self, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
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
            let mut outputs = vec![];

            for TypedShape { elem_ty, dims } in output_shapes {
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
                outputs.push(OpenclTensor {
                    tensor: Tensor::empty_of_type(elem_ty, dims),
                    buf,
                })
            }

            let kernel = match &node.op {
                Op::Add => compile_add(&context)?,
                op => todo!("{op:?}"),
            };
            execution_plans.push(ExecutionPlan { kernel, node_id });
        }

        // let buffer =
        // memory::create_buffer(context.get(), flags, count * mem::size_of::<T>(), host_ptr)?;
        // let a = unsafe {
        // memory::create_buffer(context.get(), CL_MEM_READ_ONLY, 4 * 100, ptr::null_mut())
        // }
        // .unwrap();
        // let mut sums = vec![1.0; 100];
        // let _event = unsafe {
        //     enqueue_read_buffer(
        //         queue.get(),
        //         a,
        //         CL_BLOCKING,
        //         0,
        //         4 * 100,
        //         sums.as_mut_ptr() as *mut _,
        //         0,
        //         ptr::null(),
        //     )
        //     .unwrap()
        // };
        // pub unsafe fn enqueue_read_buffer<T>(
        //     &self,
        //     buffer: &Buffer<T>,
        //     blocking_read: cl_bool,
        //     offset: size_t,
        //     data: &mut [T],
        //     event_wait_list: &[cl_event],
        // ) -> Result<Event> {
        //     let event = enqueue_read_buffer(
        //         self.queue,
        //         buffer.get(),
        //         blocking_read,
        //         offset,
        //         (data.len() * mem::size_of::<T>()) as size_t,
        //         data.as_mut_ptr() as cl_mem,
        //         event_wait_list.len() as cl_uint,
        //         if !event_wait_list.is_empty() {
        //             event_wait_list.as_ptr()
        //         } else {
        //             ptr::null()
        //         },
        //     )?;
        //     Ok(Event::new(event))
        // Ok(Event::new(event))

        // TODO: Actual compilation performs here...

        Ok(OpenclSession {
            model,
            device,
            context,
            queue,
            values,
            execution_plans,
        })

        //
        //             // Create a command_queue on the Context's device
        //             let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        //                 .expect("CommandQueue::create_default failed");
        //
        //             // Build the OpenCL program source and create the kernel.
        //             let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        //                 .expect("Program::create_and_build_from_source failed");
        //             let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");
        //
        //             /////////////////////////////////////////////////////////////////////
        //             // Compute data
        //
        //             // The input data
        //             const ARRAY_SIZE: usize = 1 << 29;
        //             // let ones: [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
        //             let ones = vec![1.0; ARRAY_SIZE];
        //             let mut sums = vec![0.0; ARRAY_SIZE];
        //             let start = Instant::now();
        //             for i in 0..ARRAY_SIZE {
        //                 sums[i] = 1.0 + 1.0 * i as cl_float;
        //             }
        //             println!("cpu: {:?}", start.elapsed());
        //
        //             // Create OpenCL device buffers
        // let mut x = unsafe {
        //     Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
        // };
        //             let mut y = unsafe {
        //                 Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
        //             };
        //             let z = unsafe {
        //                 Buffer::<cl_float>::create(
        //                     &context,
        //                     CL_MEM_WRITE_ONLY,
        //                     ARRAY_SIZE,
        //                     ptr::null_mut(),
        //                 )?
        //             };
        //
        //             // Blocking write
        //             let _x_write_event =
        //                 unsafe { queue.enqueue_write_buffer(&mut x, cl_blocking, 0, &ones, &[])? };
        //
        //             // non-blocking write, wait for y_write_event
        //             let y_write_event =
        //                 unsafe { queue.enqueue_write_buffer(&mut y, cl_non_blocking, 0, &sums, &[])? };
        //
        //             // a value for the kernel function
        //             let a: cl_float = 300.0;
        //
        //             // use the executekernel builder to set the kernel buffer and
        //             // cl_float value arguments, before setting the one dimensional
        //             // global_work_size for the call to enqueue_nd_range.
        //             // unwraps the result to get the kernel execution event.
        //             let kernel_event = unsafe {
        //                 executekernel::new(&kernel)
        //                     .set_arg(&z)
        //                     .set_arg(&x)
        //                     .set_arg(&y)
        //                     .set_arg(&a)
        //                     .set_global_work_size(array_size)
        //                     .set_wait_event(&y_write_event)
        //                     .enqueue_nd_range(&queue)?
        //             };
        //
        //             let mut events: vec<cl_event> = vec::default();
        //             events.push(kernel_event.get());
        //
        //             // create a results array to hold the results from the opencl device
        //             // and enqueue a read command to read the device buffer into the array
        //             // after the kernel event completes.
        //             // let mut results: [cl_float; array_size] = [0.0; array_size];
        //             let mut results = vec![0.0; array_size];
        //             let start = instant::now();
        //             let read_event = unsafe {
        //                 queue.enqueue_read_buffer(&z, cl_non_blocking, 0, &mut results, &events)?
        //             };
        //
        //             // wait for the read_event to complete.
        //             read_event.wait()?;
        //             println!("gpu {:?}", start.elapsed());
        //
        //             // output the first and last results
        //             println!("results front: {}", results[0]);
        //             println!("results back: {}", results[array_size - 1]);
        //
        //             // calculate the kernel duration, from the kernel_event
        //             let start_time = kernel_event.profiling_command_start()?;
        //             let end_time = kernel_event.profiling_command_end()?;
        //             let duration = end_time - start_time;
        //             println!(
        //                 "kernel execution duration (ms): {}",
        //                 duration as f64 / 1000. / 1000.0
        //             );
        //
        //             ok(())
        //         }
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
    use altius_core::{
        node::{Node, Op},
        tensor::{TensorElemType, TypedShape},
    };

    if !is_opencl_supported() {
        return;
    }

    let mut model = Model::default();

    let in_0 = model.values.new_val_named_and_shaped(
        "x".to_owned(),
        TypedShape::new(vec![8, 8].into(), TensorElemType::F32),
    );
    let in_1 = model.values.new_val_named_and_shaped(
        "y".to_owned(),
        TypedShape::new(vec![8, 8].into(), TensorElemType::F32),
    );
    let out = model.values.new_val_named_and_shaped(
        "z".to_owned(),
        TypedShape::new(vec![8, 8].into(), TensorElemType::F32),
    );
    model.add_node(Node::new(Op::Add).with_ins(vec![in_0, in_1]).with_out(out));

    model.inputs.push(in_0);
    model.inputs.push(in_1);
    model.outputs.push(out);

    let sess = OpenclSessionBuilder::new()
        .with_model(&model)
        .build()
        .unwrap();

    let x = Tensor::rand::<f32>(vec![8, 8].into());
    let y = Tensor::rand::<f32>(vec![8, 8].into());
    let z = sess.run(vec![(in_0, x), (in_1, y)]);

    println!("{:?}", z);
}
