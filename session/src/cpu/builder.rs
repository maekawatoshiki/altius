use std::{
    fs::{self, File},
    io::{BufWriter, Write as _},
    path::PathBuf,
};

use altius_core::{
    analysis::shape::infer_shapes,
    model::Model,
    node::NodeId,
    op::{Conv2d, Flatten, Gemm, HardSigmoid, MaxPool, Op},
    tensor::{Tensor, TypedShape},
    value::ValueId,
};
use indent::indent_all_by;
use rustc_hash::FxHashMap;
use thread_local::ThreadLocal;

use crate::{create_execution_plan, NodeExecutionPlan, SessionError};

#[cfg(feature = "cuda")]
use super::session::SafeCudnnContext;
use super::{session::CPUSession, thread::ThreadCtx};
#[cfg(feature = "cuda")]
use cudnn::CudnnContext;

pub struct CPUSessionBuilder {
    model: Model,
    intra_op_num_threads: usize,
    enable_profiling: bool,
}

impl CPUSessionBuilder {
    pub const fn new(model: Model) -> Self {
        Self {
            model,
            intra_op_num_threads: 1,
            enable_profiling: false,
        }
    }

    pub const fn with_intra_op_num_threads(mut self, intra_op_num_threads: usize) -> Self {
        self.intra_op_num_threads = intra_op_num_threads;
        self
    }

    pub const fn with_profiling_enabled(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    pub fn build(self) -> Result<CPUSession, SessionError> {
        let sorted_nodes = self.model.topo_sort_nodes();
        let mut inferred_shapes = FxHashMap::default();
        let mut value_shapes = FxHashMap::default();
        infer_shapes(
            &self.model,
            &sorted_nodes,
            &mut inferred_shapes,
            &mut value_shapes,
        )?;

        let execution_plans = create_execution_plan(&self.model, &sorted_nodes);

        Translator::new(&self.model, &inferred_shapes, &value_shapes)?
            .translate_into_c(&execution_plans)?;

        Ok(CPUSession {
            execution_plans,
            model: self.model,
            inferred_shapes,
            enable_profiling: self.enable_profiling,
            values: ThreadLocal::new(),
            dummy_value: Tensor::zeros::<f32>(vec![0].into()),
            tctx: ThreadCtx::new_with_num_threads(self.intra_op_num_threads),
        })
    }
}

struct Translator<'a> {
    model: &'a Model,
    inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    created_kernels: Vec<String>,
    created_values: Vec<String>,
    created_calls: Vec<String>,
    created_file_paths: Vec<PathBuf>,
    tempdir: tempfile::TempDir,
    save_generated_files: bool,
}

impl<'a> Translator<'a> {
    fn new(
        model: &'a Model,
        inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
        value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            model,
            inferred_shapes,
            value_shapes,
            created_kernels: Vec::new(),
            created_values: Vec::new(),
            created_calls: Vec::new(),
            created_file_paths: Vec::new(),
            tempdir: tempfile::tempdir()?,
            save_generated_files: true,
        })
    }

    fn translate_into_c(
        &mut self,
        execution_plans: &[NodeExecutionPlan],
    ) -> Result<(), SessionError> {
        let main_file = self.create_file("main.c")?;
        let mut writer = BufWriter::new(main_file);

        for plan in execution_plans {
            self.translate_node(plan.node_id)?;
        }

        for (&id, shape) in self.value_shapes {
            let name = self.value_name(id);
            self.created_values.push(format!(
                "float *{name} = (float *)malloc(sizeof(float) * {});",
                if shape.dims.is_scalar() {
                    "1".to_string()
                } else {
                    shape
                        .dims
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(" * ")
                }
            ));
        }

        {
            let headers = format!(
                "#include <assert.h>
#include <blis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>\n\n"
            );
            writer.write_all(headers.as_bytes())?;

            for kernel in &self.created_kernels {
                writer.write_all(kernel.as_bytes())?;
                writer.write_all(b"\n\n")?;
            }

            writer.write_all(b"int main() {\n")?;

            for value in &self.created_values {
                writer.write_all(b"    ")?;
                writer.write_all(value.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"\n")?;

            for call in &self.created_calls {
                writer.write_all(b"    ")?;
                writer.write_all(call.as_bytes())?;
                writer.write_all(b"\n")?;
            }

            writer.write_all(b"}\n")?;
        }

        writer.flush()?;

        if self.save_generated_files {
            let dir = "/tmp/model";
            let _ = fs::remove_dir_all(dir); // Ignore errors if the directory does not exist.
            fs::create_dir(dir)?;
            for path in &self.created_file_paths {
                let name = path.file_name().unwrap();
                fs::copy(path, PathBuf::from(dir).join(name)).unwrap();
            }
        }

        Ok(())
    }

    fn translate_node(&mut self, node_id: NodeId) -> Result<(), SessionError> {
        let node = &self.model.nodes[node_id];
        let inputs = node
            .inputs
            .iter()
            .map(|value_id| &self.value_shapes[value_id])
            .collect::<Vec<_>>();
        let (op, outputs) = self.inferred_shapes.get(&node_id).map_or_else(
            || todo!("Why is this node output shape not inferred?"),
            |result| Ok::<&(Op, Vec<TypedShape>), SessionError>(result),
        )?;

        let node_name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("{}_noname_{}", node.op.name(), node_id.index()));
        let node_name = escape_name(node_name);
        log::debug!("Translating node: {}", node_name);

        let args = node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .map(|id| self.value_name(*id))
            .collect::<Vec<_>>();
        self.created_calls
            .push(format!("{node_name}({});", args.join(", ")));

        match op {
            Op::Conv2d(ref c) => self.translate_conv2d(c, node_name, args, &inputs, &outputs)?,
            Op::HardSigmoid(ref h) => self.translate_hard_sigmoid(h, &inputs, &outputs)?,
            Op::Add => self.translate_add(&inputs, &outputs)?,
            Op::Mul => self.translate_mul(&inputs, &outputs)?,
            Op::ReLU => self.translate_relu(&inputs, &outputs)?,
            Op::GlobalAveragePool => self.translate_gavg_pool(&inputs, &outputs)?,
            Op::MaxPool(ref m) => self.translate_max_pool(m, &inputs, &outputs)?,
            Op::Flatten(ref f) => self.translate_flatten(f, &inputs, &outputs)?,
            Op::Gemm(ref g) => self.translate_gemm(g, &inputs, &outputs)?,
            _ => todo!("Translation not implemented for {:?}", op),
        };

        Ok(())
    }

    fn translate_conv2d(
        &mut self,
        op: &Conv2d,
        name: String,
        args: Vec<String>,
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];
        log::debug!("input names: {:?}", input_names);
        log::debug!("output names: {:?}", output_names);

        let input = &inputs[Op::CONV2D_IN];
        let weight = &inputs[Op::CONV2D_WEIGHT];
        let output = &outputs[0];

        let kernel = &op.kernel_shape;
        let padding = &op.padding;
        let stride = &op.strides;
        let dilations = &op.dilations;

        let batch_size = input.dims[0]; // .dims()[0];
        let input_c = input.dims[1];
        let input_h = input.dims[2];
        let input_w = input.dims[3];
        let input_hw = input_h * input_w;
        let output_h = output.dims[2];
        let output_w = output.dims[3];
        let output_hw = output_h * output_w;
        let _dilation = 1;
        let group = op.group as usize;
        let in_c_per_g = input_c / group;
        let out_c_per_g = output.dims[1] / group;
        let stride_h = stride[0];
        let stride_w = stride[1];
        let dilation_h = dilations[0];
        let dilation_w = dilations[1];
        let kernel_h = kernel[0];
        let kernel_w = kernel[1];

        assert_eq!(dilations.len(), 2);
        assert!(padding.len() == 4);
        let pad_t = padding[0];
        let pad_l = padding[1];
        let _pad_b = padding[2];
        let _pad_r = padding[3];

        log::debug!("kernel: {:?}", kernel);

        let code_fill_bias = if let Some(bias) = input_names.get(Op::CONV2D_BIAS) {
            let output_name = &output_names[0];
            format!(
                "{{
    float *output_ptr = {output_name};
    for (int b = 0; b < {batch_size}; b++) {{
        float *bias_ptr = (float *){bias};
        for (int g = 0; g < {group}; g++) {{
            for (int oc = 0; oc < {out_c_per_g}; oc++) {{
                for (int oh = 0; oh < {output_h}; oh++) {{
                    for (int ow = 0; ow < {output_w}; ow++) {{
                        *output_ptr = *bias_ptr;
                        output_ptr++;
                    }}
                    bias_ptr++;
                }}
            }}
        }}
    }}
}}"
            )
        } else {
            "{{ }}".to_string()
        };

        let kernel = format!(
            "static void {name}({}) {{
{}
}}",
            input_names
                .iter()
                .map(|name| format!("const float *{}", name))
                .chain(output_names.iter().map(|name| format!("float *{}", name)))
                .collect::<Vec<_>>()
                .join(", "),
            indent_all_by(4, code_fill_bias)
        );
        // log::debug!("kernel: {}", kernel);
        self.created_kernels.push(kernel);

        Ok(())
    }

    fn translate_hard_sigmoid(
        &mut self,
        _hs: &HardSigmoid,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_add(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_mul(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_relu(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_gavg_pool(
        &mut self,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_max_pool(
        &mut self,
        _max_pool: &MaxPool,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_flatten(
        &mut self,
        _flatten: &Flatten,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn translate_gemm(
        &mut self,
        _gemm: &Gemm,
        _inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<(), SessionError> {
        Ok(())
    }

    fn create_file(&mut self, name: &str) -> Result<File, SessionError> {
        let path = self.tempdir.path().join(name);
        let file = fs::File::create(&path)?;
        self.created_file_paths.push(path);
        Ok(file)
    }

    fn value_name(&self, id: ValueId) -> String {
        let value = &self.model.values.inner()[id];
        escape_name(
            value
                .name
                .clone()
                .unwrap_or_else(|| format!("Value_noname_{}", id.index())),
        )
    }
}

fn escape_name(s: impl Into<String>) -> String {
    s.into()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}
