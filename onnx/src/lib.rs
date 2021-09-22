use altius_core::{
    self as core, dim::Dimensions, model::Model, node::NodeBuilder, tensor::TensorData,
};
use prost::Message;
use rustc_hash::FxHashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/prost/onnx.rs"));

pub fn load_onnx_model<P: AsRef<Path>>(filename: P) -> Model {
    let mut f = File::open(&filename).expect("file not found");
    let meta = fs::metadata(&filename).expect("unable to read metadata");
    let mut buf = vec![0; meta.len() as usize];
    f.read_exact(&mut buf).expect("buffer overflow");
    let onnx_model = ModelProto::decode(buf.as_slice()).expect("failed to decode buf");
    let graph = onnx_model.graph.as_ref().unwrap();

    let mut name_to_node_id = FxHashMap::default();

    let mut model = Model::new();

    for node in &graph.node {
        let node_id = match node.op_type.as_str() {
            "Conv" => model.new(core::node::Conv2d::default().into()),
            "Add" => model.new(core::node::Add::default().into()),
            "Relu" => model.new(core::node::Relu::default().into()),
            "MaxPool" => model.new(core::node::MaxPool::default().into()),
            "Reshape" => model.new(core::node::Reshape::default().into()),
            "MatMul" => model.new(core::node::MatMul::default().into()),
            _ => todo!(),
        };
        name_to_node_id.insert(node.name.to_owned(), node_id);
        for output in &node.output {
            name_to_node_id.insert(output.to_owned(), node_id);
        }
    }

    for init in &graph.initializer {
        let dims = core::dim::Dimensions(init.dims.iter().map(|&d| d as usize).collect());
        let node_id = model.new({
            let t = core::tensor::Tensor::new(dims);
            if init.float_data.is_empty() {
                t.into()
            } else {
                t.with_data(TensorData::new_raw(init.float_data.to_owned()))
                    .into()
            }
        });
        name_to_node_id.insert(init.name.to_owned(), node_id);
    }

    for node in &graph.node {
        let mut inputs = vec![];
        let mut inputs_dims = vec![];
        for input in &node.input {
            if !name_to_node_id.contains_key(input) {
                let dims = get_valinfo_dims(graph, input).unwrap();
                let input_node = model.new(core::node::Node::Input(dims));
                name_to_node_id.insert(input.clone(), input_node);
            }
            inputs.push(name_to_node_id[input]);
            inputs_dims.push(get_valinfo_dims(graph, input).unwrap());
        }
        let output_dims = get_valinfo_dims(graph, node.output[0].as_str()).unwrap();
        match node.op_type.as_str() {
            "Conv" => {
                assert!(inputs.len() == 2);
                let conv = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_conv2d_mut()
                    .unwrap();
                let kernel = get_attr_dims(&node.attribute, "kernel_shape").unwrap();
                let strides = get_attr_dims(&node.attribute, "strides").unwrap();
                conv.input_node = Some(inputs[0]);
                conv.weight_node = Some(inputs[1]);
                conv.output_dims = output_dims;
                conv.input_dims = inputs_dims[0].clone();
                conv.weight_dims = inputs_dims[1].clone();
                conv.kernel = kernel;
                conv.stride = strides;
                conv.padding = vec![2, 2].into();
            }
            "Add" => {
                assert!(inputs.len() == 2);
                let add = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_add_mut()
                    .unwrap();
                add.input_a_dims = inputs_dims[0].clone();
                add.input_b_dims = inputs_dims[1].clone();
                add.input_a_node = Some(inputs[0]);
                add.input_b_node = Some(inputs[1]);
                add.output_dims = output_dims;
            }
            "Relu" => {
                let relu = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_relu_mut()
                    .unwrap();
                relu.input_node = Some(inputs[0]);
                relu.input_dims = inputs_dims[0].clone();
            }
            "MaxPool" => {
                let max_pool = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_max_pool_mut()
                    .unwrap();
                max_pool.input_node = Some(inputs[0]);
                max_pool.input_dims = inputs_dims[0].clone();
                max_pool.output_dims = output_dims;
                max_pool.kernel = get_attr_dims(&node.attribute, "kernel_shape").unwrap();
                max_pool.stride = get_attr_dims(&node.attribute, "strides").unwrap();
            }
            "Reshape" => {
                let reshape = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_reshape_mut()
                    .unwrap();
                reshape.input_node = Some(inputs[0]);
                reshape.input_dims = inputs_dims[0].clone();
                reshape.output_dims = output_dims;
            }
            "MatMul" => {
                let mat_mul = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_mat_mul_mut()
                    .unwrap();
                mat_mul.input_a_node = Some(inputs[0]);
                mat_mul.input_b_node = Some(inputs[1]);
                mat_mul.input_a_dims = inputs_dims[0].clone();
                mat_mul.input_b_dims = inputs_dims[1].clone();
                mat_mul.output_dims = output_dims;
            }
            _ => todo!(),
        };
    }

    model.output_node = Some(name_to_node_id["Plus214"]);

    model
}

fn get_attr_dims(attrs: &[AttributeProto], name: &str) -> Option<Dimensions> {
    for attr in attrs {
        if attr.name == name {
            return Some(
                attr.ints
                    .iter()
                    .map(|d| *d as usize)
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
    }
    None
}

fn get_valinfo_dims(graph: &GraphProto, name: &str) -> Option<Dimensions> {
    for &valinfo in &[&graph.input, &graph.output, &graph.value_info] {
        for v in valinfo {
            if v.name == name {
                let type_proto::Value::TensorType(t) =
                    v.r#type.as_ref().unwrap().value.as_ref().unwrap();
                let type_proto::Tensor { shape, .. } = t;
                let mut dims = vec![];
                for dim in &shape.as_ref().unwrap().dim {
                    if let tensor_shape_proto::dimension::Value::DimValue(d) =
                        dim.value.as_ref().unwrap()
                    {
                        dims.push(*d as usize);
                    }
                }
                return Some(dims.into());
            }
        }
    }
    None
}
