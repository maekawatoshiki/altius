use altius_core::{
    self as core, dim::Dimensions, model::Model, node::NodeBuilder, shape_inferer::infer_shapes,
    tensor::TensorData,
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
    let mut initializers = FxHashMap::default();
    // let mut input_names=Fx

    let mut model = Model::new();

    // Register empty nodes first
    for node in &graph.node {
        let node_id = model.new(match node.op_type.as_str() {
            "Conv" => core::node::Conv2d::default().into(),
            "Add" => core::node::Add::default().into(),
            "Relu" => core::node::Relu::default().into(),
            "MaxPool" => core::node::MaxPool::default().into(),
            "Reshape" => core::node::Reshape::default().into(),
            "MatMul" => core::node::MatMul::default().into(),
            op => todo!("unsupported operation: {}", op),
        });
        name_to_node_id.insert(node.name.to_owned(), node_id);
        for output in &node.output {
            name_to_node_id.insert(output.to_owned(), node_id);
        }
    }

    // Register an input node
    {
        let input_name = graph.input[0].name.clone();
        let dims = get_valinfo_dims(graph, input_name.as_str()).unwrap();
        let input_node = model.new(core::node::Node::Input(dims));
        name_to_node_id.insert(input_name, input_node);
    }

    for init in &graph.initializer {
        if init.float_data.is_empty() {
            initializers.insert(init.name.to_owned(), init);
            continue;
        }
        let dims = core::dim::Dimensions(init.dims.iter().map(|&d| d as usize).collect());
        let node_id = model.new(
            core::tensor::Tensor::new(dims)
                .with_data(TensorData::new_raw(init.float_data.to_owned()))
                .into(),
        );
        name_to_node_id.insert(init.name.to_owned(), node_id);
    }

    for node in &graph.node {
        let mut inputs = vec![];
        for input in &node.input {
            if !name_to_node_id.contains_key(input) {
                continue;
            }
            inputs.push(name_to_node_id[input]);
        }
        match node.op_type.as_str() {
            "Conv" => {
                assert!(inputs.len() == 2);
                let conv = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_conv2d_mut()
                    .unwrap();
                let kernel = get_attr_dims(&node.attribute, "kernel_shape").unwrap();
                let strides = get_attr_dims(&node.attribute, "strides").unwrap();
                conv.inputs = [Some(inputs[0]), Some(inputs[1])];
                conv.kernel = kernel;
                conv.stride = strides;
                conv.padding = vec![2usize, 2].into();
            }
            "Add" => {
                assert!(inputs.len() == 2);
                let add = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_add_mut()
                    .unwrap();
                add.inputs = [Some(inputs[0]), Some(inputs[1])];
            }
            "Relu" => {
                let relu = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_relu_mut()
                    .unwrap();
                relu.input = Some(inputs[0]);
            }
            "MaxPool" => {
                let max_pool = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_max_pool_mut()
                    .unwrap();
                max_pool.input = Some(inputs[0]);
                max_pool.kernel = get_attr_dims(&node.attribute, "kernel_shape").unwrap();
                max_pool.stride = get_attr_dims(&node.attribute, "strides").unwrap();
            }
            "Reshape" => {
                let reshape = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_reshape_mut()
                    .unwrap();
                let shape = initializers[&node.input[1]];
                assert!(!shape.int64_data.is_empty());
                reshape.input = Some(inputs[0]);
                reshape.output_dims = shape
                    .int64_data
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>()
                    .into();
            }
            "MatMul" => {
                let mat_mul = model.arena_mut()[name_to_node_id[&node.name]]
                    .as_mat_mul_mut()
                    .unwrap();
                mat_mul.inputs = [Some(inputs[0]), Some(inputs[1])];
            }
            _ => todo!(),
        };
    }

    model.output_node = Some(name_to_node_id[&graph.output[0].name]);

    infer_shapes(&mut model);

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
