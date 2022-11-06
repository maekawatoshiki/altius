// use std::borrow::Cow;
// use wgpu::util::DeviceExt;

use std::{borrow::Cow, time::Instant};

use altius_core::{
    model::Model,
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
use rustc_hash::FxHashMap;
use tera::Context;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BufferUsages, Instance,
};

use crate::{interpreter::infer_shapes, SessionError};

/// See <https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsperdimension>
const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u64 = 65535;

/// See <https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsizex>
const MAX_WORKGROUP_SIZE_X: u64 = 256;
// const MAX_WORKGROUP_SIZE_Y: u64 = 256;

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

        let shape = [1, 10, 1024, 1024];
        let input_a = Tensor::rand::<f32>(shape.to_vec().into());
        let input_b = Tensor::rand::<f32>(shape.to_vec().into());
        let mut output = Tensor::zeros::<f32>(shape.to_vec().into());
        compute_add(&device, &queue, &input_a, &input_b, &mut output).await;
        compute_add(&device, &queue, &input_a, &input_b, &mut output).await;
        compute_add(&device, &queue, &input_a, &input_b, &mut output).await;
        let start = Instant::now();
        assert!(input_a
            .data::<f32>()
            .iter()
            .zip(input_b.data::<f32>().iter())
            .zip(output.data::<f32>())
            .all(|((a, b), &o)| a + b == o));
        println!("cpu elapsed: {:?}", start.elapsed());

        Ok(vec![])
    }
}

fn compile(model: &Model, device: &wgpu::Device, queue: &wgpu::Queue) {
    let sorted_nodes = model.topo_sort_nodes();
    let mut shapes = FxHashMap::default();
    infer_shapes(model, &sorted_nodes, &mut shapes);

    for &node_id in &sorted_nodes {
        let node = &model.nodes[node_id];
    }
}

fn compile_add(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_0: &TypedShape,
    input_1: &TypedShape,
    output: &TypedShape,
    bind_group: &wgpu::BindGroup,
) {
    let wgsl = r#"
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, write> output: array<f32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    output[i] = input_0[i] {{ op }} input_1[i];
}
        "#;
    let Some((num_threads_x, workgroup_size_x)) = workgroup_size(
        output.dims.total_elems() as u64,
        MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        MAX_WORKGROUP_SIZE_X,
    ) else {
        todo!("Too many elements")
    };
    let mut template_ctx = Context::new();
    template_ctx.insert("workgroup_size_x", &workgroup_size_x);
    template_ctx.insert("op", "+");
    let wgsl = tera::Tera::one_off(wgsl, &template_ctx, true).ok().unwrap();

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for _ in 0..1 {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(num_threads_x, 1, 1);
    }
}

async fn compute_add(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_a: &Tensor,
    input_b: &Tensor,
    output: &mut Tensor,
) {
    let wgsl = r#"
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, write> output: array<f32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    output[i] = input_0[i] {{ op }} input_1[i];
}
        "#;

    let Some((num_threads_x, workgroup_size_x)) = workgroup_size(
        output.dims().total_elems() as u64,
        MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        MAX_WORKGROUP_SIZE_X,
    ) else {
        todo!()
    };
    let mut template_ctx = Context::new();
    template_ctx.insert("workgroup_size_x", &workgroup_size_x);
    template_ctx.insert("op", "+");
    let wgsl = tera::Tera::one_off(wgsl, &template_ctx, true).ok().unwrap();

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
    });

    println!("{num_threads_x}, {workgroup_size_x}");

    let input_a = input_a.data::<f32>();
    let input_b = input_b.data::<f32>();
    let output = output.data_mut::<f32>();

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

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for _ in 0..1 {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(num_threads_x, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_output_buf, 0, size);

    let buffer_slice = staging_output_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    // Submits command encoder for processing
    let start = Instant::now();
    queue.submit(Some(encoder.finish()));
    queue.on_submitted_work_done(move || {
        log::info!("wgpu work elapsed: {:?}", start.elapsed());
    });

    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
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

fn workgroup_size(x: u64, max_threads: u64, max_workgroup_size: u64) -> Option<(u32, u32)> {
    Some(if x > max_threads as u64 {
        let workgroup_size = x.div_ceil(max_threads as u64);
        let threads = x.div_ceil(workgroup_size as u64);

        if threads > max_threads {
            return None;
        }

        if workgroup_size > max_workgroup_size {
            return None;
        }

        (threads as u32, workgroup_size as u32)
    } else {
        (x as u32, 1)
    })
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
