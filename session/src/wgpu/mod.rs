// use std::borrow::Cow;
// use wgpu::util::DeviceExt;

use std::borrow::Cow;

use altius_core::{model::Model, tensor::Tensor, value::ValueId};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BufferUsages, Instance,
};

use crate::SessionError;

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

    pub const fn model(&self) -> &Model {
        self.model
    }

    pub const fn wgpu_instance(&self) -> &Instance {
        &self.instance
    }

    pub async fn run(&self, _inputs: Vec<(ValueId, Tensor)>) -> Result<Vec<Tensor>, SessionError> {
        let adapter = self
            .instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or(SessionError::Message("Failed to request adapter.".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let _info = adapter.get_info();

        // skip this on LavaPipe temporarily
        // if info.vendor == 0x10005 {
        //     return None;
        // }

        let input_a = vec![1f32];
        let input_b = vec![2f32];
        let mut output = vec![0f32];
        compute_add(&device, &queue, &input_a, &input_b, &mut output).await;
        assert_eq!(output[0], 3f32);
        // println!("result {:?}", result);

        Ok(vec![])
    }
}

async fn compute_add(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
) {
    let wgsl = r#"
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, write> output: array<f32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    output[i] = input_a[i] + input_b[i];
}
        "#;
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });

    assert_eq!(input_a.len(), input_b.len());
    assert_eq!(input_a.len(), output.len());

    // Gets the size in bytes of the buffer.
    let slice_size = input_a.len() * std::mem::size_of::<f32>();
    let size = slice_size as wgpu::BufferAddress;

    let input_a_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(input_a),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let input_b_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(input_b),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size,
        // usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size,
        // usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute add");
        cpass.dispatch_workgroups(output.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_output_buf, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_output_buf.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        // let result = bytemuck::cast_slice(&data).to_vec();
        // let result = bytemuck::cast_slice(&data).to_vec();
        output.copy_from_slice(bytemuck::cast_slice(&data));

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_output_buf.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory
    } else {
        panic!("failed to run compute on gpu!")
    }
}

#[test]
fn instantiate() {
    let model = Model::default();
    let _ = WgpuSession::new(&model);
}

#[test]
fn add() {
    let model = Model::default();
    let sess = WgpuSession::new(&model);
    pollster::block_on(async {
        sess.run(vec![]).await.unwrap();
    });
}
