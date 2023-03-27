use std::{
    fs::{self, create_dir_all, remove_file, File},
    io::{BufWriter, Write as _},
    path::{Path, PathBuf},
};

use altius_core::{
    analysis::shape::infer_shapes,
    model::Model,
    node::{Node, NodeId},
    op::{
        Conv2d, Flatten, Gemm, HardSigmoid, LayerNormalization, MaxPool, Op, ReduceMean, Softmax,
        Transpose,
    },
    tensor::{TensorElemType, TypedShape},
    value::ValueId,
};
use indent::indent_all_by;
use rustc_hash::{FxHashMap, FxHashSet};
use sha1::{Digest, Sha1};

use crate::{create_execution_plan, NodeExecutionPlan, SessionError};

use super::session::CPUSession;

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

        let mut profile_symbols = FxHashMap::default();
        let mut translator = Translator::new(&self.model, &inferred_shapes, &value_shapes)?
            .with_profiling_enabled(self.enable_profiling);
        {
            translator.translate_into_c(&execution_plans)?;
            translator.compile()?;
        }
        let lib = unsafe { libloading::Library::new(translator.target_dir.join("model.so")) }?;
        let entry: libloading::Symbol<unsafe extern "C" fn()> = unsafe { lib.get(b"model_entry")? };
        let entry = unsafe { entry.into_raw().into_raw() };

        for (&val_id, tensor) in &self.model.inits {
            let name = translator.value_name(val_id);
            let entry: libloading::Symbol<*const *const u8> = unsafe { lib.get(name.as_bytes())? };
            unsafe { *entry.cast_mut() = tensor.data_as_ptr() };
        }

        if self.enable_profiling {
            for name in translator.used_op_names {
                let symbol: libloading::Symbol<*const f64> =
                    unsafe { lib.get(format!("elapsed_{}", name).as_bytes())? };
                profile_symbols.insert(name, unsafe { *symbol.into_raw() });
            }
        }

        Ok(CPUSession {
            target_dir: translator.target_dir,
            model: self.model,
            lib,
            value_shapes,
            entry,
            enable_profiling: self.enable_profiling,
            profile_symbols,
        })
    }
}

struct Translator<'a> {
    model: &'a Model,
    inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
    value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    created_kernels: Vec<String>,
    created_extern_values: Vec<String>,
    created_tmp_values: Vec<String>,
    created_calls: Vec<String>,
    created_file_paths: Vec<PathBuf>,
    used_op_names: FxHashSet<String>,
    target_dir: PathBuf,
    enable_profiling: bool,
    prev_code_hash: Option<[u8; 20]>,
}

impl<'a> Translator<'a> {
    fn new(
        model: &'a Model,
        inferred_shapes: &'a FxHashMap<NodeId, (Op, Vec<TypedShape>)>,
        value_shapes: &'a FxHashMap<ValueId, TypedShape>,
    ) -> Result<Self, SessionError> {
        let target_dir = PathBuf::from("/tmp/model");
        let mut prev_code_hash = None;
        if target_dir.as_path().exists() {
            prev_code_hash = get_file_sha1("/tmp/model/main.c");
            let _ = remove_file("/tmp/model/main.c");
        }
        create_dir_all(&target_dir)?;

        Ok(Self {
            model,
            inferred_shapes,
            value_shapes,
            created_kernels: Vec::new(),
            created_extern_values: Vec::new(),
            created_tmp_values: Vec::new(),
            created_calls: Vec::new(),
            created_file_paths: Vec::new(),
            used_op_names: FxHashSet::default(),
            target_dir,
            enable_profiling: false,
            prev_code_hash,
        })
    }

    fn with_profiling_enabled(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    fn compile(&self) -> Result<(), SessionError> {
        let newer_hash = get_file_sha1("/tmp/model/main.c");
        if newer_hash.is_some() && newer_hash == self.prev_code_hash {
            return Ok(());
        }

        let mut cmd = std::process::Command::new("clang");

        #[cfg(target_os = "macos")]
        let args = &["-framework", "Accelerate", "-L/opt/homebrew/opt/libomp/lib"];
        #[cfg(target_os = "linux")]
        let args = &["-march=native", "-lblis"];

        cmd.arg("-O3")
            .arg("-o")
            .arg(self.target_dir.join("model.so"))
            .arg(self.target_dir.join("main.c"))
            .args(args)
            .arg("-fopenmp")
            .arg("-fvectorize")
            .arg("-shared")
            .arg("-fPIC")
            .arg("-lm")
            .current_dir(&self.target_dir);
        let status = cmd.status()?;
        if !status.success() {
            return Err(SessionError::Message("Failed to compile model".into()));
        }
        Ok(())
    }

    fn translate_into_c(
        &mut self,
        execution_plans: &[NodeExecutionPlan],
    ) -> Result<(), SessionError> {
        let main_file = self.create_file("main.c")?;
        let mut writer = BufWriter::new(main_file);

        for plan in execution_plans {
            // Allocate temporary tensors.
            for output in self.model.nodes[plan.node_id]
                .outputs
                .iter()
                .filter(|id| !self.model.outputs.contains(id))
            {
                self.created_calls.push(format!(
                    "{name} = (float *)malloc(sizeof(float) * ({size}));",
                    name = self.value_name(*output),
                    size = self.value_shapes[output].dims.total_elems(),
                ));
            }

            self.translate_node(plan.node_id)?;

            // Free tensors.
            for free in plan.free_vals.iter() {
                self.created_calls
                    .push(format!("free({});", self.value_name(*free)));
            }
        }

        for (&id, shape) in self.value_shapes {
            if (self.model.inputs.contains(&id) || self.model.outputs.contains(&id))
                && !self.model.inits.contains_key(&id)
            {
                continue;
            }

            let name = self.value_name(id);
            let ty = get_c_type(shape.elem_ty);

            if self.model.inits.contains_key(&id) {
                &mut self.created_extern_values
            } else {
                &mut self.created_tmp_values
            }
            .push(format!("{ty} *{name};"));
        }

        {
            #[cfg(target_os = "macos")]
            let blas = "#include <Accelerate/Accelerate.h>";
            #[cfg(not(target_os = "macos"))]
            let blas = "#include <blis/blis.h>";
            let headers = format!(
                "{blas}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

struct timespec now() {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}}

{profile}

",
                profile = if self.enable_profiling {
                    self.used_op_names
                        .iter()
                        .map(|name| format!("double elapsed_{} = 0.0;", name))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    String::new()
                }
            );
            writer.write_all(headers.as_bytes())?;

            for kernel in &self.created_kernels {
                writer.write_all(kernel.as_bytes())?;
                writer.write_all(b"\n\n")?;
            }

            for value in &self.created_extern_values {
                writer.write_all(b"")?;
                writer.write_all(value.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"\n")?;

            writer.write_all(
                format!(
                    "void model_entry({}) {{\n",
                    self.model
                        .inputs
                        .iter()
                        .filter(|&id| !self.model.inits.contains_key(id))
                        .map(|&id| {
                            let name = self.value_name(id);
                            format!("const float *{}", name)
                        })
                        .chain(self.model.outputs.iter().map(|&id| {
                            let name = self.value_name(id);
                            format!("float *{}", name)
                        }))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .as_bytes(),
            )?;

            if self.enable_profiling {
                for name in &self.used_op_names {
                    writer.write_all(format!("    elapsed_{} = 0.0;\n", name).as_bytes())?;
                }
                writer.write_all(b"\n")?;
            }

            for value in &self.created_tmp_values {
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
        self.used_op_names.insert(op.name().into());

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

        let kernel = match op {
            Op::Conv2d(ref c) => self.translate_conv2d(c, &args, &inputs, &outputs)?,
            Op::HardSigmoid(ref h) => self.translate_hard_sigmoid(h, &args, &inputs, &outputs)?,
            Op::Add => self.translate_bin_op("+", &args, &inputs, &outputs)?,
            Op::Sub => self.translate_bin_op("-", &args, &inputs, &outputs)?,
            Op::Mul => self.translate_bin_op("*", &args, &inputs, &outputs)?,
            Op::Div => self.translate_bin_op("/", &args, &inputs, &outputs)?,
            Op::Pow => self.translate_pow(&node, &args, &inputs, &outputs)?,
            Op::Sqrt => self.translate_sqrt(&args, &inputs, &outputs)?,
            Op::ReLU => self.translate_relu(&args, &inputs, &outputs)?,
            Op::GlobalAveragePool => self.translate_gavg_pool(&args, &inputs, &outputs)?,
            Op::MaxPool(ref m) => self.translate_max_pool(m, &args, &inputs, &outputs)?,
            Op::Reshape => self.translate_reshape(&args, &inputs, &outputs)?,
            Op::MatMul => self.translate_mat_mul(&args, &inputs, &outputs)?,
            Op::Flatten(ref f) => self.translate_flatten(f, &args, &inputs, &outputs)?,
            Op::Gemm(ref g) => self.translate_gemm(g, &args, &inputs, &outputs)?,
            Op::Transpose(ref t) => self.translate_transpose(t, &args, &inputs, &outputs)?,
            Op::Concat(_) => format!("assert(0 && \"concat\");"),
            Op::Gather(_) => format!("assert(0 && \"gather\");"),
            Op::ReduceMean(ref r) => self.translate_reduce_mean(r, &args, &inputs, &outputs)?,
            Op::Softmax(ref s) => self.translate_softmax(s, &args, &inputs, &outputs)?,
            Op::LayerNormalization(l) => self.translate_layer_norm(l, &args, &inputs, &outputs)?,
            Op::Gelu => self.translate_gelu(&args, &inputs, &outputs)?,
            _ => todo!("Translation not implemented for {:?}", op),
        };

        let kernel = format!(
            "void {node_name}({args}) {{
{start_profiling}
{body}
{end_profiling}
}}",
            args = inputs
                .iter()
                .map(|i| ("const ", *i))
                .chain(outputs.iter().map(|o| ("", o)))
                .zip(args.iter())
                .map(|((prefix, shape), name)| format!(
                    "{prefix}{ty} *{name}",
                    ty = get_c_type(shape.elem_ty)
                ))
                .collect::<Vec<_>>()
                .join(", "),
            body = indent_all_by(4, kernel),
            start_profiling = if self.enable_profiling {
                indent_all_by(4, format!("const struct timespec _start = now();"))
            } else {
                String::new()
            },
            end_profiling = if self.enable_profiling {
                indent_all_by(
                    4,
                    format!(
                        "{{
    const struct timespec _end = now();
    const double start_in_sec = (double)_start.tv_sec + (double)_start.tv_nsec / 1e9;
    const double end_in_sec = (double)_end.tv_sec + (double)_end.tv_nsec / 1e9;
    elapsed_{} += end_in_sec - start_in_sec;
}}",
                        op.name(),
                    ),
                )
            } else {
                String::new()
            },
        );
        self.created_kernels.push(kernel);

        Ok(())
    }

    fn translate_conv2d(
        &mut self,
        op: &Conv2d,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];
        log::debug!("input names: {:?}", input_names);
        log::debug!("output names: {:?}", output_names);

        let input = &inputs[Op::CONV2D_IN];
        let _weight = &inputs[Op::CONV2D_WEIGHT];
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
                const float bias = *bias_ptr;
                for (int o = 0; o < {output_h} * {output_w}; o++) {{
                    output_ptr[o] = bias;
                }}
                output_ptr += {output_h} * {output_w};
                bias_ptr++;
            }}
        }}
    }}
}}"
            )
        } else {
            let output_name = &output_names[0];
            format!(
                "{{
    for (int i = 0; i < {size}; i++) {{
        {output_name}[i] = 0.0f;
    }}
}}",
                size = output.dims.total_elems()
            )
        };

        let code_im2col = if pad_t == 0
            && pad_l == 0
            && stride_h == 1
            && stride_w == 1
            && dilation_h == 1
            && dilation_w == 1
        {
            let input_name = &input_names[0];

            format!("float *col =
    (float *)malloc(sizeof(float) * {batch_size} * {input_c} * {output_h} * {output_w} * {kernel_h} * {kernel_w});

{{
    int outer = {batch_size} * {input_c};
    float *input_ptr = (float *){input_name};
    float *col_ptr = (float *)col;

    while (outer > 0) {{
        int inner = {kernel_h} * {kernel_w};
        while (inner > 0) {{
            memcpy(col_ptr, input_ptr, sizeof(float) * {input_hw});
            col_ptr += {output_hw};
            inner -= 1;
        }}
        input_ptr += {input_hw};
        outer -= 1;
    }}
}}
")
        } else if dilation_h == 1 && dilation_w == 1 {
            let input_name = &input_names[0];

            format!("float *col =
    (float *)malloc(sizeof(float) * {batch_size} * {group} * {out_c_per_g} * {output_h} * {output_w} * {kernel_h} * {kernel_w});

{{
    const int output_hw = {output_h} * {output_w};
    float *input_ptr = (float *){input_name};
    float *col_ptr = (float *)col;
    
    for (int outer = 0; outer < {batch_size} * {input_c}; outer++) {{
        for (int fy = 0; fy < {kernel_h}; fy++) {{
            for (int fx = 0; fx < {kernel_w}; fx++) {{
                for (int oh = 0; oh < {output_h}; oh++) {{
                    float *col = &col_ptr[oh * {output_h}];
                    const int ih = fy + oh * {stride_h};

                    if ({pad_t} > ih || ih >= {input_h} + {pad_t}) {{
                        for (int i = 0; i < {output_w}; i++) {{
                            col[i] = 0.;
                        }}
                        continue;
                    }}

                    int ow = 0;
                    int iw = fx + (ow * {stride_w});
                    while (iw < {pad_l}) {{
                        col[ow] = 0.;
                        iw += {stride_w};
                        ow += 1;
                    }}

                    const int c = (ih - {pad_t}) * {input_w};
                    iw = fx + (ow * {stride_w});
                    while (iw < {input_w} + {pad_l}) {{
                        int jw = c + iw - {pad_l};
                        col[ow] = input_ptr[jw];
                        iw += {stride_w};
                        ow += 1;
                    }}

                    if (ow < {output_w}) {{
                        for (int i = ow; i < {output_w}; i++) {{
                            col[i] = 0.;
                        }}
                    }}
                }}
                col_ptr += {output_hw};
            }}
        }}
        input_ptr += {input_hw};
    }}
}}")
        } else {
            todo!();
        };

        let col_stride = in_c_per_g * kernel[0] * kernel[1] * output_h * output_w;
        let weight_stride = out_c_per_g * in_c_per_g * kernel[0] * kernel[1];
        let output_stride = out_c_per_g * output_hw;
        let k = in_c_per_g * kernel[0] * kernel[1];
        let outer = (batch_size * group) / in_c_per_g;
        let weight_name = &input_names[1];
        let output_name = &output_names[0];

        let code_gemm = format!(
            "{{
    float *weight_ptr = (float *){weight_name}; 
    float *output_ptr = (float *){output_name};
    float *col_ptr = col;
    int outer = {outer};

    do {{
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            {out_c_per_g}, {output_hw}, {k}, 1.,
            weight_ptr, {k}, col_ptr, {output_hw}, 1., output_ptr, {output_hw});

        weight_ptr += {weight_stride};
        output_ptr += {output_stride};
        col_ptr += {col_stride};
        outer--;
    }} while (outer > 0);
}}"
        );

        let kernel = format!(
            "{}

{}

{}

free(col);",
            code_fill_bias, code_im2col, code_gemm
        );

        Ok(kernel)
    }

    fn translate_hard_sigmoid(
        &mut self,
        hs: &HardSigmoid,
        args: &[String],
        inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let alpha = hs.alpha;
        let beta = hs.beta;
        let size = inputs[0].dims.total_elems();
        let kernel = format!(
            "for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = fminf(1.0, fmaxf(0.0, x * {alpha} + {beta}));
}}"
        );
        Ok(kernel)
    }

    fn translate_bin_op(
        &mut self,
        op: &str,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        let kernel = if inputs[0].dims == inputs[1].dims {
            format!(
                "for (int i = 0; i < {size}; i++) {{
    {output}[i] = {input_0}[i] {op} {input_1}[i];
}}",
                input_0 = input_names[0],
                input_1 = input_names[1],
                size = outputs[0].dims.total_elems(),
                output = output_name,
            )
        } else {
            let rank = outputs[0].dims.len();
            let input_0_strides = inputs[0]
                .dims
                .strides_for_broadcasting_to(&outputs[0].dims)
                .unwrap();
            let input_1_strides = inputs[1]
                .dims
                .strides_for_broadcasting_to(&outputs[0].dims)
                .unwrap();

            let mut kernel = String::new();
            for (i, ((odim, i0str), i1str)) in outputs[0]
                .dims
                .iter()
                .zip(input_0_strides.iter())
                .zip(input_1_strides.iter())
                .enumerate()
                .rev()
            {
                if i == rank - 1 {
                    kernel = format!(
                        "for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    *output_ptr = *input_0_ptr_{i} {op} *input_1_ptr_{i};
    input_0_ptr_{i} += {i0str};
    input_1_ptr_{i} += {i1str};
    output_ptr += 1;
}}"
                    );
                } else {
                    kernel = format!(
                        "{}for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    float *input_0_ptr_{iplus1} = input_0_ptr_{i};
    float *input_1_ptr_{iplus1} = input_1_ptr_{i};
{}
    input_0_ptr_{i} += {i0str};
    input_1_ptr_{i} += {i1str};
}}",
                        if i == 0 {
                            format!(
                                "float *input_0_ptr_0 = (float *){};
float *input_1_ptr_0 = (float *){};
float *output_ptr = {};\n",
                                input_names[0], input_names[1], output_name
                            )
                        } else {
                            "".to_string()
                        },
                        indent_all_by(4, kernel),
                        iplus1 = i + 1,
                    );
                }
            }

            kernel
        };

        Ok(kernel)
    }

    fn translate_pow(
        &mut self,
        node: &Node,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        let kernel = if inputs[0].dims == inputs[1].dims {
            format!(
                "for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = powf({input_0}[i], {input_1}[i]);
}}",
                input_0 = input_names[0],
                input_1 = input_names[1],
                size = outputs[0].dims.total_elems(),
            )
        } else if inputs[1].dims.is_scalar() {
            match self.model.inits.get(&node.inputs[1]) {
                Some(init) if init.data::<f32>()[0] == 2. => format!(
                    "for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = {input_0}[i] * {input_0}[i];
}}",
                    input_0 = input_names[0],
                    size = outputs[0].dims.total_elems(),
                ),
                _ => format!(
                    "for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = powf({input_0}[i], {input_1}[0]);
}}",
                    input_0 = input_names[0],
                    input_1 = input_names[1],
                    size = outputs[0].dims.total_elems(),
                ),
            }
        } else {
            format!(
                "assert(0 && \"TODO: in0.shape = {:?}, in1.shape = {:?}\");",
                inputs[0].dims, inputs[1].dims
            )
        };

        Ok(kernel)
    }

    fn translate_sqrt(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];

        assert_eq!(inputs[0].dims, outputs[0].dims);

        let kernel = format!(
            "for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = sqrtf({input_name}[i]);
}}",
            input_name = input_name,
            size = outputs[0].dims.total_elems(),
        );

        Ok(kernel)
    }

    fn translate_relu(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let size = inputs[0].dims.total_elems();
        let kernel = format!(
            "for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = fmaxf(0.0, x);
}}"
        );
        Ok(kernel)
    }

    fn translate_gavg_pool(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];

        let input = inputs[0];
        let output = &outputs[0];

        assert!(input.dims.len() == 4);
        assert!(output.dims.len() == 4);

        let Some(&[n, c, h, w]) = input.dims.get(0..4) else {
            return Err(SessionError::Message("Input must be four dimensions".into()))
        };
        let Some(&[isn, isc, _, _]) = input.dims.strides().get(0..4) else { panic!() };
        let area = h * w;
        let osn = output.dims.strides()[0];

        let kernel = format!(
            "for (int n = 0; n < {n}; n++) {{
    for (int c = 0; c < {c}; c++) {{
        float sum = 0.0;
        for (int h = 0; h < {h}; h++) {{
            for (int w = 0; w < {w}; w++) {{
                sum += {input_name}[n * {isn} + c * {isc} + h * {w} + w];
            }}
        }}
        {output_name}[n * {osn} + c] = sum / {area};
    }}
}}"
        );

        Ok(kernel)
    }

    fn translate_max_pool(
        &mut self,
        maxpool: &MaxPool,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];
        let input_name = &input_names[0];
        let output_name = &output_names[0];

        let input = inputs[Op::MAXPOOL_IN];
        let output = &outputs[Op::MAXPOOL_OUT];

        let kernel = &maxpool.kernel_shape;
        let stride = &maxpool.strides;

        assert!(input.dims.len() == 4);
        assert!(output.dims.len() == 4);

        let padding = &maxpool.padding;
        let batches = output.dims[0];
        let channels = output.dims[1];
        let outer = batches * channels;
        let output_h = output.dims[2];
        let output_w = output.dims[3];
        let input_h = input.dims[2];
        let input_w = input.dims[3];
        let kernel_h = kernel[0];
        let kernel_w = kernel[1];
        let stride_h = stride[0];
        let stride_w = stride[1];
        let input_hw = input_h * input_w;
        let output_hw = output_h * output_w;

        let pad_t = padding[0] as isize;
        let pad_l = padding[1] as isize;

        let kernel = format!(
            "float *input_ptr = (float *){input_name};
float *output_ptr = (float *){output_name};

for (int outer = 0; outer < {outer}; outer++) {{
    int y = -{pad_t};
    for (int ay = 0; ay < {output_h}; ay++) {{
        int x = -{pad_l};
        float *output = &output_ptr[ay * {output_w}];
        int fy_min = -y > 0 ? -y : 0;
        int fy_max = {kernel_h} < ({input_h} - y) ? {kernel_h} : ({input_h} - y);
        for (int i = 0; i < {output_w}; i++) {{
            float max = -INFINITY;
            int fx_min = -x > 0 ? -x : 0;
            int fx_max = {kernel_w} < ({input_w} - x) ? {kernel_w} : ({input_w} - x);
            for (int fy = fy_min; fy < fy_max; fy++) {{
                int oy = y + fy;
                for (int fx = fx_min; fx < fx_max; fx++) {{
                    int ox = x + fx;
                    max = fmaxf(max, input_ptr[oy * {input_w} + ox]);
                }}
            }}
            *output++ = max;
            x += {stride_w};
        }}
        y += {stride_h};
    }}
    input_ptr += {input_hw};
    output_ptr += {output_hw};
}}"
        );

        Ok(kernel)
    }

    fn translate_reshape(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        assert!(inputs[0].elem_ty.is_f32());
        let kernel = format!(
            "memcpy({output_name}, {input_name}, {size} * sizeof(float));",
            size = inputs[0].dims.total_elems()
        );
        Ok(kernel)
    }

    fn translate_mat_mul(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];

        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = &outputs[0];

        if input_0.dims.len() == 2 && input_1.dims.len() == 2 {
            assert_eq!(output.dims.len(), 2);
            let [m, _k] = input_0.dims.to_fixed_dims::<2>();
            let [k, n] = input_1.dims.to_fixed_dims::<2>();

            let kernel = format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    {m}, {n}, {k}, 1.,
    {in0}, {k}, {in1}, {n}, 0., {out}, {n});",
                in0 = input_names[0],
                in1 = input_names[1],
                out = output_names[0],
            );

            Ok(kernel)
        } else if input_0.dims.len() == 3 && input_1.dims.len() == 2 {
            let [batch, m, _k] = input_0.dims.to_fixed_dims::<3>();
            let [k, n] = input_1.dims.to_fixed_dims::<2>();

            let kernel = format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    {batchm}, {n}, {k}, 1.,
    {in0}, {k}, {in1}, {n}, 0., {out}, {n});",
                batchm = batch * m,
                in0 = input_names[0],
                in1 = input_names[1],
                out = output_names[0],
            );

            Ok(kernel)
        } else if input_0.dims.len() == 4 && input_1.dims.len() == 4 {
            let [one, batch, m, _k] = input_0.dims.to_fixed_dims::<4>();
            let [one_, batch_, k, n] = input_1.dims.to_fixed_dims::<4>();
            assert_eq!(one, 1);
            assert_eq!(one_, 1);
            assert_eq!(batch, batch_);

            let kernel = format!(
                "float *input_0_ptr = (float *){input_0};
float *input_1_ptr = (float *){input_1};
float *output_ptr = (float *){output};
for (int i = 0; i < {batch}; i++) {{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        {m}, {n}, {k}, 1.,
        input_0_ptr, {k}, input_1_ptr, {n}, 0., output_ptr, {n});
    input_0_ptr += {m} * {k};
    input_1_ptr += {k} * {n};
    output_ptr += {m} * {n};
}}",
                input_0 = input_names[0],
                input_1 = input_names[1],
                output = output_names[0],
            );

            Ok(kernel)
        } else {
            Ok(format!(
                "assert(0 && \"TODO: in0.shape={:?}, in1.shape={:?}\");",
                input_0.dims, input_1.dims
            ))
        }
    }

    fn translate_flatten(
        &mut self,
        _flatten: &Flatten,
        args: &[String],
        inputs: &[&TypedShape],
        _outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];
        assert!(inputs[0].elem_ty.is_f32());
        let kernel = format!(
            "memcpy({output_name}, {input_name}, {size} * sizeof(float));",
            size = inputs[0].dims.total_elems()
        );
        Ok(kernel)
    }

    fn translate_gemm(
        &mut self,
        gemm: &Gemm,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let input_2 = inputs[2];
        let output = &outputs[0];
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];

        assert!(input_0.dims.len() == 2);
        assert!(input_1.dims.len() == 2);
        assert!(input_2.dims.len() == 1);

        let m = input_0.dims[gemm.trans_a as usize];
        let k = input_0.dims[1 - gemm.trans_a as usize];
        let n = input_1.dims[1 - gemm.trans_b as usize];

        let kernel = format!(
            "for (int i = 0; i < {output_size}; i += {n}) {{
    memcpy({out} + i, {in2}, {n} * sizeof(float));
}}
cblas_sgemm(CblasRowMajor, {transa}, {transb},
    {m}, {n}, {k}, 1.,
    {in0}, {lda}, {in1}, {ldb}, 1., {out}, {n});",
            transa = if gemm.trans_a {
                "CblasTrans"
            } else {
                "CblasNoTrans"
            },
            transb = if gemm.trans_b {
                "CblasTrans"
            } else {
                "CblasNoTrans"
            },
            output_size = output.dims.total_elems(),
            in0 = input_names[0],
            in1 = input_names[1],
            in2 = input_names[2],
            lda = if gemm.trans_a { m } else { k },
            ldb = if gemm.trans_b { k } else { n },
            out = output_names[0]
        );

        Ok(kernel)
    }

    fn translate_transpose(
        &mut self,
        transpose: &Transpose,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];
        let input = inputs[0];
        let output = &outputs[0];

        assert!(input.elem_ty.is_f32());
        assert!(output.elem_ty.is_f32());
        assert_eq!(input.dims.len(), output.dims.len());

        struct PermIter {
            num_axes: usize,
            index: Vec<usize>,
            upper_bound: Vec<usize>,
            stride: Vec<usize>,
        }

        fn next_index(perm: &mut PermIter, mut index: usize) -> usize {
            let mut pos = perm.num_axes - 1;
            index += perm.stride[pos];
            perm.index[pos] += 1;
            if perm.index[pos] < perm.upper_bound[pos] {
                return index;
            }

            index -= perm.stride[pos] * perm.index[pos];
            perm.index[pos] = 0;
            if pos == 0 {
                return index;
            }

            loop {
                pos -= 1;
                index += perm.stride[pos];
                perm.index[pos] += 1;
                if perm.index[pos] < perm.upper_bound[pos] {
                    break;
                }
                index -= perm.stride[pos] * perm.index[pos];
                perm.index[pos] = 0;
                if pos == 0 {
                    break;
                }
            }
            index
        }

        let in_dims = &input.dims;
        let in_strides = input.dims.strides();
        let out_dims = &output.dims;
        let num_axes = in_dims.len();
        let new_strides = transpose
            .perm
            .iter()
            .map(|&axis| in_strides[axis as usize])
            .collect::<Vec<_>>();

        let mut num_blocks = 1;
        let mut num_elems_in_block = 1;
        let mut suffix = true;
        let mut reduced_num_axes = 0;

        for i in (0..num_axes).rev() {
            let input_axis = transpose.perm[i] as usize;
            if suffix && input_axis == i {
                num_elems_in_block *= in_dims[input_axis];
            } else {
                suffix = false;
                num_blocks *= in_dims[input_axis];
                reduced_num_axes += 1;
            }
        }

        let mut perm = PermIter {
            num_axes: reduced_num_axes,
            index: vec![0; reduced_num_axes],
            upper_bound: out_dims[0..reduced_num_axes].to_vec(),
            stride: new_strides[0..reduced_num_axes].to_vec(),
        };

        let mut src_idx = 0;
        let mut indices = vec![];
        for _ in 0..num_blocks {
            indices.push(src_idx);
            src_idx = next_index(&mut perm, src_idx);
        }
        let indices = indices
            .iter()
            .map(|&idx| idx.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let kernel = if num_elems_in_block == 1 {
            format!("int src_indices[{num_blocks}] = {{ {indices} }};
for (int i = 0; i < {num_blocks}; i++) {{
    {out}[i] = {in}[src_indices[i]];
}}",
                out = output_name,
                in = input_name,
                indices = indices
            )
        } else if num_blocks == 1 {
            format!(
                "memcpy({}, {}, sizeof(float) * {});",
                output_name, input_name, num_elems_in_block
            )
        } else {
            format!("int src_indices[{num_blocks}] = {{ {indices} }};
for (int i = 0; i < {num_blocks}; i++) {{
    int src_idx = src_indices[i];
    memcpy({out} + i * {num_elems_in_block},
            {in} + src_indices[i],
            sizeof(float) * {num_elems_in_block});
}}",
                out = output_name,
                in = input_name,
                num_blocks = num_blocks,
                num_elems_in_block = num_elems_in_block,
                indices = indices
            )
        };

        Ok(kernel)
    }

    fn translate_reduce_mean(
        &mut self,
        rmean: &ReduceMean,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input = inputs[0];
        let output = &outputs[0];
        let input_name = &args[0];
        let output_name = &args[1];

        let axes = rmean
            .axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (input.dims.len() as i64 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(axes.len(), 1);

        let axis = axes[0];
        assert_eq!(input.dims.len(), 3);
        assert_eq!(axis, 2);
        assert!(rmean.keep_dims);

        let axis_len = input.dims[2];
        let r_axis_len = 1.0 / axis_len as f32;

        let kernel = format!(
            "for (int i = 0; i < {batch}; i++) {{ 
    float sum = 0.0;
    for (int j = 0; j < {axis_len}; j++) {{
        sum += {input_name}[i * {axis_len} + j];
    }}
    {output_name}[i] = sum * {r_axis_len};
}}",
            batch = output.dims.total_elems()
        );

        Ok(kernel)
    }

    fn translate_softmax(
        &mut self,
        softmax: &Softmax,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input = inputs[0];
        let output = &outputs[0];
        let input_name = &args[0];
        let output_name = &args[1];

        assert!(softmax.axis == -1 || softmax.axis == (input.dims.len() - 1) as i64);

        let axis_len = *input.dims.last().unwrap();

        let kernel = format!(
            "for (int i = 0; i < {batch}; i++) {{
    float sum = 0.0;
    for (int j = 0; j < {axis_len}; j++) {{
        {output_name}[i * {axis_len} + j] = exp({input_name}[i * {axis_len} + j]);
        sum += exp({input_name}[i * {axis_len} + j]);
    }}
    float recip_sum = 1.0 / sum;
    for (int j = 0; j < {axis_len}; j++) {{
        {output_name}[i * {axis_len} + j] = {output_name}[i * {axis_len} + j] * recip_sum;
    }}
}}",
            batch = output.dims.total_elems() / axis_len,
        );

        Ok(kernel)
    }

    fn translate_layer_norm(
        &mut self,
        ln: &LayerNormalization,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let data = inputs[0];
        let _scale = inputs[1];
        let _bias = inputs[2];
        let output = &outputs[0];
        let data_name = &args[0];
        let scale_name = &args[1];
        let bias_name = &args[2];
        let output_name = &args[inputs.len()..][0];

        assert!(
            ln.axis == -1 || ln.axis == *data.dims.last().unwrap() as i64,
            "Axis must be the last dimension."
        );
        assert!(ln.stash_type == 1, "Stash type must be 1.");
        assert!(data.elem_ty.is_f32(), "Input data type must be f32.");
        assert_eq!(data.dims.total_elems(), output.dims.total_elems());

        let axis_len = *data.dims.last().unwrap();

        let kernel = format!(
"for (int i = 0; i < {batch}; i++) {{
    float sum = 0.0;
    for (int j = 0; j < {axis_len}; j++) {{
        sum += {data_name}[i * {axis_len} + j];
    }}
    float mean = sum * {inv_axis_len};
    for (int j = 0; j < {axis_len}; j++) {{
        {output_name}[i * {axis_len} + j] = {data_name}[i * {axis_len} + j] - mean;
    }}
    float sum_squares = 0.0;
    for (int j = 0; j < {axis_len}; j++) {{
        const float x = {output_name}[i * {axis_len} + j];
        sum_squares += x * x;
    }}
    float inv_mean = 1.0 / sqrtf(sum_squares * {inv_axis_len} + {epsilon});
    for (int j = 0; j < {axis_len}; j++) {{
        {output_name}[i * {axis_len} + j] = {output_name}[i * {axis_len} + j] * inv_mean * {scale_name}[j] + {bias_name}[j];
    }}
}}",
            batch = data.dims.total_elems() / axis_len,
            inv_axis_len = (axis_len as f32).recip(),
            epsilon = ln.epsilon,
        );

        Ok(kernel)
    }

    fn translate_gelu(
        &mut self,
        args: &[String],
        inputs: &[&TypedShape],
        outputs: &[TypedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];
        let _input = inputs[0];
        let output = &outputs[0];

        const B: f32 = 0.7978845608028654f32; // sqrt(2.0 / PI)
        const C: f32 = 0.035677408136300125f32; // 0.044715 * sqrt(2.0 / PI)

        let kernel = format!(
            "for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = 0.5 * x * (1.0 + tanhf(x * ({C} * x * x + {B})));
}}",
            size = output.dims.total_elems(),
        );

        Ok(kernel)
    }

    fn create_file(&mut self, name: &str) -> Result<File, SessionError> {
        let path = self.target_dir.join(name);
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
    fn add_prefix(s: String) -> String {
        if s.starts_with(|c: char| c.is_ascii_digit()) {
            format!("_{}", s)
        } else {
            s.to_string()
        }
    }
    add_prefix(
        s.into()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect(),
    )
}

fn get_file_sha1(path: impl AsRef<Path>) -> Option<[u8; 20]> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = Sha1::new();
    std::io::copy(&mut file, &mut hasher).ok()?;
    let hash = hasher.finalize();
    hash.get(..20)?.try_into().ok()
}

fn get_c_type(ty: TensorElemType) -> &'static str {
    match ty {
        TensorElemType::F32 => "float",
        TensorElemType::I32 => "int32_t",
        TensorElemType::I64 => "int64_t",
        TensorElemType::Bool => "unsigned char",
    }
}