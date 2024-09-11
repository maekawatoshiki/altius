use std::path::Path;

use altius_core::{
    onnx::load_onnx,
    optimize::{gelu_fusion::fuse_gelu, layer_norm_fusion::fuse_layer_norm},
    tensor::Tensor,
};
use altius_session_interpreter::InterpreterSessionBuilder;
use criterion::{criterion_group, criterion_main, Criterion};

const THREADS: usize = 1;

fn without_gelu(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let model = load_onnx(root.join("deit.onnx"))
        .expect("Failed to load model. Have you run altius_py/deit.py?");

    let input = Tensor::rand::<f32>(vec![1, 3, 224, 224].into());

    let sess = InterpreterSessionBuilder::new(model)
        .with_intra_op_num_threads(THREADS)
        .build()
        .unwrap();
    c.bench_function("No fusion", |b| b.iter(|| sess.run(vec![input.clone()])));
}

fn with_gelu(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut model = load_onnx(root.join("deit.onnx"))
        .expect("Failed to load model. Have you run altius_py/deit.py?");
    fuse_gelu(&mut model);

    let input = Tensor::rand::<f32>(vec![1, 3, 224, 224].into());

    let sess = InterpreterSessionBuilder::new(model)
        .with_intra_op_num_threads(THREADS)
        .build()
        .unwrap();
    c.bench_function("Gelu fusion", |b| b.iter(|| sess.run(vec![input.clone()])));
}

fn with_gelu_ln(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut model = load_onnx(root.join("deit.onnx"))
        .expect("Failed to load model. Have you run altius_py/deit.py?");
    fuse_layer_norm(&mut model);
    fuse_gelu(&mut model);

    let input = Tensor::rand::<f32>(vec![1, 3, 224, 224].into());

    let sess = InterpreterSessionBuilder::new(model)
        .with_intra_op_num_threads(THREADS)
        .build()
        .unwrap();
    c.bench_function("LN fusion, Gelu fusion", |b| {
        b.iter(|| sess.run(vec![input.clone()]))
    });
}

fn with_gelu_ln2(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut model = load_onnx(root.join("deit.onnx"))
        .expect("Failed to load model. Have you run altius_py/deit.py?");
    fuse_gelu(&mut model);
    fuse_layer_norm(&mut model);

    let input = Tensor::rand::<f32>(vec![1, 3, 224, 224].into());

    let sess = InterpreterSessionBuilder::new(model)
        .with_intra_op_num_threads(THREADS)
        .build()
        .unwrap();
    c.bench_function("Gelu fusion, LN fusion", |b| {
        b.iter(|| sess.run(vec![input.clone()]))
    });
}

criterion_group! {
    fusion,
    with_gelu,
    with_gelu_ln,
    with_gelu_ln2,
    without_gelu,
}

criterion_main!(fusion);
