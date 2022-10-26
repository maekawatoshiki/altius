// use std::borrow::Cow;
// use wgpu::util::DeviceExt;

use altius_core::model::Model;
use wgpu::{Backends, Instance};

pub struct WgpuSession<'a> {
    model: &'a Model,
    instance: Instance,
}

impl<'a> WgpuSession<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            instance: Instance::new(Backends::all()),
        }
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    pub fn wgpu_instance(&self) -> &Instance {
        &self.instance
    }
}

#[test]
fn instantiate() {
    let model = Model::default();
    let _ = WgpuSession::new(&model);
}

// // Indicates a u32 overflow in an intermediate Collatz value
// const OVERFLOW: u32 = 0xffffffff;
//
// #[test]
// fn collatz() {
//     pollster::block_on(run());
// }
//
// async fn run() {
//     let numbers = vec![1, 2, 3, 4];
//
//     let steps = execute_gpu(&numbers).await.unwrap();
//
//     let disp_steps: Vec<String> = steps
//         .iter()
//         .map(|&n| match n {
//             OVERFLOW => "OVERFLOW".to_string(),
//             _ => n.to_string(),
//         })
//         .collect();
//
//     println!("Steps: [{}]", disp_steps.join(", "));
//     #[cfg(target_arch = "wasm32")]
//     log::info!("Steps: [{}]", disp_steps.join(", "));
// }
//
// async fn execute_gpu(numbers: &[u32]) -> Option<Vec<u32>> {
//     // Instantiates instance of WebGPU
//     let instance = wgpu::Instance::new(wgpu::Backends::all());
//
//     // `request_adapter` instantiates the general connection to the GPU
//     let adapter = instance
//         .request_adapter(&wgpu::RequestAdapterOptions::default())
//         .await?;
//
//     // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
//     //  `features` being the available features.
//     let (device, queue) = adapter
//         .request_device(
//             &wgpu::DeviceDescriptor {
//                 label: None,
//                 features: wgpu::Features::empty(),
//                 limits: wgpu::Limits::downlevel_defaults(),
//             },
//             None,
//         )
//         .await
//         .unwrap();
//
//     let info = adapter.get_info();
//
//     // skip this on LavaPipe temporarily
//     if info.vendor == 0x10005 {
//         return None;
//     }
//
//     execute_gpu_inner(&device, &queue, numbers).await
// }
//
// async fn execute_gpu_inner(
//     device: &wgpu::Device,
//     queue: &wgpu::Queue,
//     numbers: &[u32],
// ) -> Option<Vec<u32>> {
//     let wgsl = r#"
// @group(0)
// @binding(0)
// var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience
//
// // The Collatz Conjecture states that for any integer n:
// // If n is even, n = n/2
// // If n is odd, n = 3n+1
// // And repeat this process for each new n, you will always eventually reach 1.
// // Though the conjecture has not been proven, no counterexample has ever been found.
// // This function returns how many times this recurrence needs to be applied to reach 1.
// fn collatz_iterations(n_base: u32) -> u32{
//     var n: u32 = n_base;
//     var i: u32 = 0u;
//     loop {
//         if (n <= 1u) {
//             break;
//         }
//         if (n % 2u == 0u) {
//             n = n / 2u;
//         }
//         else {
//             // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
//             if (n >= 1431655765u) {   // 0x55555555u
//                 return 4294967295u;   // 0xffffffffu
//             }
//
//             n = 3u * n + 1u;
//         }
//         i = i + 1u;
//     }
//     return i;
// }
//
// @compute
// @workgroup_size(1)
// fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
// }
//         "#;
//     // Loads the shader from WGSL
//     let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
//         label: None,
//         source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
//     });
//
//     // Gets the size in bytes of the buffer.
//     let slice_size = numbers.len() * std::mem::size_of::<u32>();
//     let size = slice_size as wgpu::BufferAddress;
//
//     // Instantiates buffer without data.
//     // `usage` of buffer specifies how it can be used:
//     //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
//     //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
//     let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//         label: None,
//         size,
//         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//         mapped_at_creation: false,
//     });
//
//     // Instantiates buffer with data (`numbers`).
//     // Usage allowing the buffer to be:
//     //   A storage buffer (can be bound within a bind group and thus available to a shader).
//     //   The destination of a copy.
//     //   The source of a copy.
//     let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("Storage Buffer"),
//         contents: bytemuck::cast_slice(&numbers),
//         usage: wgpu::BufferUsages::STORAGE
//             | wgpu::BufferUsages::COPY_DST
//             | wgpu::BufferUsages::COPY_SRC,
//     });
//
//     // A bind group defines how buffers are accessed by shaders.
//     // It is to WebGPU what a descriptor set is to Vulkan.
//     // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
//
//     // A pipeline specifies the operation of a shader
//
//     // Instantiates the pipeline.
//     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         label: None,
//         layout: None,
//         module: &cs_module,
//         entry_point: "main",
//     });
//
//     // Instantiates the bind group, once again specifying the binding of buffers.
//     let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
//     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//         label: None,
//         layout: &bind_group_layout,
//         entries: &[wgpu::BindGroupEntry {
//             binding: 0,
//             resource: storage_buffer.as_entire_binding(),
//         }],
//     });
//
//     // A command encoder executes one or many pipelines.
//     // It is to WebGPU what a command buffer is to Vulkan.
//     let mut encoder =
//         device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//     {
//         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
//         cpass.set_pipeline(&compute_pipeline);
//         cpass.set_bind_group(0, &bind_group, &[]);
//         cpass.insert_debug_marker("compute collatz iterations");
//         cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
//     }
//     // Sets adds copy operation to command encoder.
//     // Will copy data from storage buffer on GPU to staging buffer on CPU.
//     encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
//
//     // Submits command encoder for processing
//     queue.submit(Some(encoder.finish()));
//
//     // Note that we're not calling `.await` here.
//     let buffer_slice = staging_buffer.slice(..);
//     // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
//     let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
//     buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
//
//     // Poll the device in a blocking manner so that our future resolves.
//     // In an actual application, `device.poll(...)` should
//     // be called in an event loop or on another thread.
//     device.poll(wgpu::Maintain::Wait);
//
//     // Awaits until `buffer_future` can be read from
//     if let Some(Ok(())) = receiver.receive().await {
//         // Gets contents of buffer
//         let data = buffer_slice.get_mapped_range();
//         // Since contents are got in bytes, this converts these bytes back to u32
//         // let result = bytemuck::cast_slice(&data).to_vec();
//         let result = bytemuck::cast_slice(&data).to_vec();
//
//         // With the current interface, we have to make sure all mapped views are
//         // dropped before we unmap the buffer.
//         drop(data);
//         staging_buffer.unmap(); // Unmaps buffer from memory
//                                 // If you are familiar with C++ these 2 lines can be thought of similarly to:
//                                 //   delete myPointer;
//                                 //   myPointer = NULL;
//                                 // It effectively frees the memory
//
//         // Returns data from buffer
//         Some(result)
//     } else {
//         panic!("failed to run compute on gpu!")
//     }
// }
