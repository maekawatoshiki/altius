use rustc_hash::FxHashMap;

use crate::{
    analysis::shape::{ShapeError, infer_shapes},
    model::Model,
    op::Op,
};

pub fn compute_flops(model: &Model) -> Result<usize, ShapeError> {
    let nodes = model.topo_sort_nodes(); // TODO: Dead node elimination
    let mut inferred_shapes = FxHashMap::default();
    let mut value_shapes = FxHashMap::default();
    infer_shapes(model, &mut inferred_shapes, &mut value_shapes)?;
    let mut flops = 0;
    for node_id in nodes {
        let node = &model.graph.nodes[node_id];
        flops += match &node.op {
            Op::MatMul => {
                let a_shape = &value_shapes[&node.inputs[0]];
                let b_shape = &value_shapes[&node.inputs[1]];
                let m = a_shape.dims[a_shape.dims.len() - 2];
                let k = a_shape.dims[a_shape.dims.len() - 1];
                let n = b_shape.dims[b_shape.dims.len() - 1];
                let rem = a_shape.dims[..a_shape.dims.len() - 2]
                    .iter()
                    .product::<usize>();
                2 * rem * m * n * k
            }
            Op::Conv2d(c) => {
                let input_shape = &value_shapes[&node.inputs[0]];
                let kernel_shape = &value_shapes[&node.inputs[1]];
                let output_shape = &value_shapes[&node.outputs[0]];
                output_shape.dims.total_elems()
                    * (input_shape.dims[1] / c.group as usize
                        * kernel_shape.dims[2..].iter().product::<usize>())
                    * (1 + (node.inputs.len() == 3) as usize)
            }
            Op::Gemm(_) => {
                let a_shape = &value_shapes[&node.inputs[0]];
                let b_shape = &value_shapes[&node.inputs[1]];
                assert_eq!(a_shape.dims.len(), 2);
                assert_eq!(b_shape.dims.len(), 2);
                let m = a_shape.dims[0];
                let k = a_shape.dims[1];
                let n = b_shape.dims[1];
                2 * m * n * k + 3 * m * n
            }
            _ => 0,
        };
    }
    Ok(flops)
}

#[test]
fn test_compute_flops() {
    let model = Model::default();
    let flops = compute_flops(&model).unwrap();
    assert_eq!(flops, 0);
}
