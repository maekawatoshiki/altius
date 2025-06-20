use std::{
    collections::BTreeMap,
    fs::{create_dir_all, remove_file, File},
    io::{BufWriter, Write},
    mem::{size_of, ManuallyDrop},
    ops::Range,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use altius_core::{
    model::Model,
    node::{Node, NodeId},
    op::{
        BatchNormalization, Cast, Concat, Conv2d, Flatten, FusedActivation, FusedElemwise, Gather,
        Gemm, HardSigmoid, LayerNormalization, MaxPool, Op, ReduceMax, ReduceMean, Resize, Softmax,
        Split, Transpose,
    },
    tensor::{TensorElemType, TypedFixedShape},
    value::ValueId,
};
use altius_session::{plan::create_execution_plan, SessionError};
use cranelift::prelude::{InstBuilder, IntCC, Type, Variable};
use cranelift::{
    codegen::settings::Configurable,
    prelude::{FunctionBuilder, FunctionBuilderContext},
};
use cranelift_codegen::{
    entity::EntityRef,
    ir::{
        self,
        types::{F32, I32, I64, I8},
        AbiParam, Function, MemFlags, Value,
    },
    Context,
};
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use indent::indent_all_by;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use sha1::{Digest, Sha1};
use target_lexicon::Triple;

pub(super) struct Translator<'a> {
    model: &'a Model,
    inferred_shapes: &'a HashMap<NodeId, (Op, Vec<TypedFixedShape>)>,
    value_shapes: &'a HashMap<ValueId, TypedFixedShape>,
    created_kernels: Vec<String>,
    created_kernel_protos: Vec<String>,
    reshaped_values: HashSet<ValueId>,
    propagated_inits: HashSet<ValueId>,
    used_op_names: HashSet<String>,
    target_dir: PathBuf,
    enable_profiling: bool,
    intra_op_num_threads: usize,
    prev_code_hash: Option<[u8; 20]>,
    clif_ctx: CraneliftCtx,
    enable_clif: bool,
}

pub(super) struct TranslationProduct<'a> {
    pub model: &'a Model,
    pub used_op_names: HashSet<String>,
    pub target_dir: PathBuf,
}

struct CraneliftCtx {
    ctx: Context,
    module: ObjectModule,
    num_vars: usize,
}

/// Manages memory regions for temporary tensors.
#[derive(Default)]
struct Regions {
    start_to_region: BTreeMap<usize, Range<usize>>,
    val_to_start: HashMap<ValueId, usize>,
    max_allocated: usize,
}

impl<'a> Translator<'a> {
    pub fn new(
        model: &'a Model,
        inferred_shapes: &'a HashMap<NodeId, (Op, Vec<TypedFixedShape>)>,
        value_shapes: &'a HashMap<ValueId, TypedFixedShape>,
    ) -> Result<Self, SessionError> {
        let target_dir = std::env::var("ALTIUS_MODEL_OUT_DIR").map_or_else(
            |_| Ok::<_, SessionError>(tempfile::TempDir::new()?.keep()),
            |dir| Ok(PathBuf::from(dir)),
        )?;
        let mut prev_code_hash = None;
        if target_dir.as_path().exists() {
            let files = glob::glob(target_dir.join("*.c").as_path().to_str().unwrap())
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>();
            prev_code_hash = compute_sha1_from_files(&files);
            for f in files {
                let _ = remove_file(f);
            }
        }
        create_dir_all(&target_dir)?;

        Ok(Self {
            model,
            inferred_shapes,
            value_shapes,
            created_kernels: Vec::new(),
            created_kernel_protos: Vec::new(),
            reshaped_values: HashSet::default(),
            propagated_inits: HashSet::default(),
            used_op_names: HashSet::default(),
            target_dir,
            enable_profiling: false,
            intra_op_num_threads: 1,
            prev_code_hash,
            clif_ctx: CraneliftCtx::default(),
            enable_clif: {
                // TODO: In the future when we no longer depend on C compiler to compile the model,
                //       we should remove this environment variable.
                let val = std::env::var("ALTIUS_ENABLE_CLIF").unwrap_or_else(|_| "0".to_string());
                assert!(
                    val == "0" || val == "1",
                    "ALTIUS_ENABLE_CLIF must be 0 or 1"
                );
                val == "1"
            },
        })
    }

    pub fn with_profiling_enabled(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    pub fn with_intra_op_num_threads(mut self, intra_op_num_threads: usize) -> Self {
        self.intra_op_num_threads = intra_op_num_threads;
        self
    }

    pub fn compile(mut self) -> Result<TranslationProduct<'a>, SessionError> {
        log::debug!("Compiling the model...");

        self.translate_into_c()?;

        // TODO: For cranelift
        // TODO: Clean this code
        let mut objects = Vec::new();
        if self.enable_clif {
            let product = self.clif_ctx.module.finish();
            let filename = self.target_dir.join("clif.o");
            File::create(&filename)?.write_all(product.emit().unwrap().as_slice())?;
            objects.push(filename);
        }

        let new_hash = compute_sha1_from_files(
            &glob::glob(self.target_dir.join("*.c").as_path().to_str().unwrap())
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
        );
        if new_hash.is_some() && new_hash == self.prev_code_hash {
            log::debug!("Skipped compiling!");
            return Ok(TranslationProduct {
                model: self.model,
                used_op_names: self.used_op_names,
                target_dir: self.target_dir,
            });
        }

        let mut cmd = std::process::Command::new("clang");

        #[cfg(debug_assertions)]
        let mimalloc_path = "target/debug/build/libmimalloc-sys-*/out/*-static.o";
        #[cfg(not(debug_assertions))]
        let mimalloc_path = "target/release/build/libmimalloc-sys-*/out/*-static.o";
        #[cfg(target_os = "linux")]
        #[cfg(debug_assertions)]
        let blis_path = "target/debug/build/blis-src-*/out";
        #[cfg(target_os = "linux")]
        #[cfg(not(debug_assertions))]
        let blis_path = "target/release/build/blis-src-*/out";
        // TODO: Remove unwraps.
        let mimalloc_obj = find_path_from_project_root(mimalloc_path).unwrap();
        #[cfg(target_os = "linux")]
        let blis_path = find_path_from_project_root(blis_path).unwrap();

        #[cfg(target_os = "macos")]
        let args = &[
            "-framework",
            "Accelerate",
            "-fno-math-errno",
            "-L/opt/homebrew/opt/libomp/lib",
            mimalloc_obj.to_str().unwrap(),
        ];
        #[cfg(target_os = "linux")]
        let args = &["-march=native", "-lblis", mimalloc_obj.to_str().unwrap()];

        let num_compilied_kernels = Arc::new(AtomicUsize::new(0));
        let num_kernels_to_compile = self.created_kernels.len();
        objects.append(
            &mut self
                .created_kernels
                .chunks(self.created_kernels.len() / num_cpus::get() + 1)
                .enumerate()
                .map(|(i, chunks)| {
                    let num_kernels = chunks.len();
                    let target_dir = self.target_dir.clone();
                    #[cfg(target_os = "linux")]
                    let blis_include_dir = blis_path.join("include").to_str().unwrap().to_string();
                    let num_compilied_kernels = num_compilied_kernels.clone();
                    let thread = std::thread::spawn(move || -> Result<(), SessionError> {
                        let mut cmd = std::process::Command::new("clang");
                        cmd.arg("-O3")
                            .arg("-c")
                            .arg("-o")
                            .arg(target_dir.join(format!("kernels-{i:05}.o")))
                            .arg(target_dir.join(format!("kernels-{i:05}.c")))
                            .arg("-fno-math-errno")
                            .arg("-fopenmp")
                            .arg("-fvectorize")
                            .arg("-fPIC");
                        #[cfg(target_os = "macos")]
                        cmd.args(["-DACCELERATE_NEW_LAPACK", "-DACCELERATE_LAPACK_ILP64"]);
                        #[cfg(target_os = "linux")]
                        cmd.arg("-march=native")
                            .arg(format!("-I{}", blis_include_dir));
                        if !cmd.status()?.success() {
                            return Err(SessionError::Message(
                                "Failed to compile the model".into(),
                            ));
                        }
                        num_compilied_kernels.fetch_add(num_kernels, Ordering::SeqCst);
                        eprint!(
                            "[{:03}/{:03}] Compilation in progress\r",
                            num_compilied_kernels.load(Ordering::SeqCst),
                            num_kernels_to_compile
                        );
                        Ok(())
                    });
                    (thread, self.target_dir.join(format!("kernels-{i:05}.o")))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(t, file)| {
                    t.join().unwrap()?;
                    Ok(file)
                })
                .collect::<Result<Vec<_>, SessionError>>()?,
        );

        cmd.arg("-O3")
            .arg("-o")
            .arg(self.target_dir.join("model.so"))
            .arg(self.target_dir.join("main.c"))
            .args(objects)
            .args(args)
            .arg("-fopenmp")
            .arg("-fvectorize")
            .arg("-shared")
            .arg("-fPIC")
            .arg("-lm");
        #[cfg(target_os = "linux")]
        for (flag, name) in [("-I", "include"), ("-L", "lib")].iter() {
            cmd.arg(format!(
                "{}{}",
                flag,
                blis_path.join(name).to_str().unwrap()
            ));
        }
        let status = cmd.status()?;
        if !status.success() {
            return Err(SessionError::Message("Failed to compile model".into()));
        }

        log::debug!("Finished compiling the model.");

        Ok(TranslationProduct {
            model: self.model,
            used_op_names: self.used_op_names,
            target_dir: self.target_dir,
        })
    }

    fn translate_into_c(&mut self) -> Result<(), SessionError> {
        let main_file = self.create_file("main.c")?;
        let mut writer = BufWriter::new(main_file);

        let execution_plans = create_execution_plan(self.model);

        let mut created_calls = vec![];
        let mut created_tmp_values = vec![];
        let mut created_extern_values = vec![];
        let mut regions = Regions::default();

        for plan in execution_plans {
            let node = &self.model.graph.nodes[plan.node_id];
            // Allocate temporary tensors.
            for output in node
                .outputs
                .iter()
                .filter(|id| !self.model.graph.outputs.contains(id))
            {
                if matches!(node.op, Op::Reshape | Op::Squeeze(_) | Op::Unsqueeze(_)) {
                    continue;
                }
                let shape = &self.value_shapes[output];
                let ty = get_c_type(shape.elem_ty);
                let name = self.value_name(*output);
                let size = shape.dims.total_elems() * shape.elem_ty.size();
                let region = regions.alloc(*output, size);
                let offset = region.start;
                created_calls.push(format!(
                    "{name} = ({ty} *)((char *)global_memory + {offset});",
                ));
            }

            self.translate_node(plan.node_id, &mut created_calls)?;

            // Free tensors.
            for free in plan.free_vals.iter() {
                if self.reshaped_values.contains(free) || self.propagated_inits.contains(free) {
                    continue;
                }
                regions.free(*free);
            }
        }

        for (&id, shape) in self.value_shapes {
            if (self.model.graph.inputs.contains(&id) || self.model.graph.outputs.contains(&id))
                && !self.model.graph.inits.contains_key(&id)
            {
                continue;
            }

            let name = self.value_name(id);
            let ty = get_c_type(shape.elem_ty);

            if self.model.graph.inits.contains_key(&id) {
                &mut created_extern_values
            } else {
                &mut created_tmp_values
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

#define malloc mi_malloc
#define free mi_free

void *mi_malloc(size_t size);
void mi_free(void *ptr);
void *mi_malloc_aligned(size_t size, size_t alignment);

static struct timespec now() {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}}\n\n",
            );
            writer.write_all(headers.as_bytes())?;
            writer.write_all(b"static char *global_memory;\n\n")?;
            if self.enable_profiling {
                writer.write_all(
                    format!(
                        "{profile}\n\n",
                        profile = self
                            .used_op_names
                            .iter()
                            .map(|name| format!("double elapsed_{name};"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                    .as_bytes(),
                )?;
            }

            for (i, chunk) in self
                .created_kernels
                .chunks(self.created_kernels.len() / num_cpus::get() + 1)
                .enumerate()
            {
                let main_file = self.create_file(format!("kernels-{i:05}.c").as_str())?;
                let mut writer = BufWriter::new(main_file);
                writer.write_all(headers.as_bytes())?;
                if self.enable_profiling {
                    writer.write_all(
                        format!(
                            "{profile}\n\n",
                            profile = self
                                .used_op_names
                                .iter()
                                .map(|name| format!("extern double elapsed_{name};"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        )
                        .as_bytes(),
                    )?;
                }
                for kernel in chunk {
                    writer.write_all(kernel.as_bytes())?;
                    writer.write_all(b"\n\n")?;
                }
                writer.flush()?;
            }

            for proto in &self.created_kernel_protos {
                writer.write_all(proto.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"\n")?;

            for value in created_extern_values {
                writer.write_all(b"")?;
                writer.write_all(value.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"\n")?;

            #[cfg(target_os = "macos")]
            let init_blis = "// blis not used on macOS";
            #[cfg(not(target_os = "macos"))]
            let init_blis = format!("bli_thread_set_num_threads({});", self.intra_op_num_threads);
            writer.write_all(
                format!(
                    r#"void initialize() {{
    {init_blis}
    global_memory = (char *)mi_malloc_aligned({max_allocated}, 32);
}}

"#,
                    max_allocated = regions.max_allocated
                )
                .as_bytes(),
            )?;

            writer.write_all(
                format!(
                    "void model_entry({}) {{\n\n",
                    self.model
                        .graph
                        .inputs
                        .iter()
                        .filter(|&id| !self.model.graph.inits.contains_key(id))
                        .map(|&id| {
                            let name = self.value_name(id);
                            let shape = &self.value_shapes[&id];
                            let ty = get_c_type(shape.elem_ty);
                            format!("const {ty} *{name}")
                        })
                        .chain(self.model.graph.outputs.iter().map(|&id| {
                            let name = self.value_name(id);
                            let shape = &self.value_shapes[&id];
                            let ty = get_c_type(shape.elem_ty);
                            format!("{ty} *{name}")
                        }))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .as_bytes(),
            )?;

            if self.enable_profiling {
                for name in &self.used_op_names {
                    writer.write_all(format!("    elapsed_{name} = 0.0;\n").as_bytes())?;
                }
                writer.write_all(b"\n")?;
            }

            for value in created_tmp_values {
                writer.write_all(b"    ")?;
                writer.write_all(value.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"\n")?;

            for call in created_calls {
                writer.write_all(indent_all_by(4, call).as_bytes())?;
                writer.write_all(b"\n")?;
            }

            writer.write_all(b"}\n\n")?;

            writer.write_all(
                format!(
                    "void trampoline(const void **ins, const void **outs) {{
    model_entry({args});
}}\n",
                    args = self
                        .model
                        .graph
                        .inputs
                        .iter()
                        .filter(|&id| !self.model.graph.inits.contains_key(id))
                        .enumerate()
                        .map(|(i, &id)| {
                            let shape = &self.value_shapes[&id];
                            let ty = get_c_type(shape.elem_ty);
                            format!("(const {ty} *)ins[{i}]")
                        })
                        .chain(self.model.graph.outputs.iter().enumerate().map(|(i, &id)| {
                            let shape = &self.value_shapes[&id];
                            let ty = get_c_type(shape.elem_ty);
                            format!("({ty} *)outs[{i}]")
                        }))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .as_bytes(),
            )?;
        }

        writer.flush()?;

        Ok(())
    }

    fn translate_node(
        &mut self,
        node_id: NodeId,
        created_calls: &mut Vec<String>,
    ) -> Result<(), SessionError> {
        let node = &self.model.graph.nodes[node_id];
        let inputs = node
            .inputs
            .iter()
            .map(|value_id| &self.value_shapes[value_id])
            .collect::<Vec<_>>();
        let (op, outputs) = self.inferred_shapes.get(&node_id).map_or_else(
            || todo!("Why is this node output shape not inferred?"),
            Ok::<&(Op, Vec<TypedFixedShape>), SessionError>,
        )?;
        self.used_op_names.insert(op.name().into());

        let node_name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("{}_noname_{}", node.op.name(), node_id.index()));
        let node_name = escape_name(node_name);
        // log::debug!("Translating node: {}", node_name);

        let args = node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .map(|id| self.value_name(*id))
            .collect::<Vec<_>>();

        if matches!(op, Op::Reshape | Op::Squeeze(_) | Op::Unsqueeze(_)) {
            // TODO: Support 'Flatten'.
            self.reshaped_values.insert(node.inputs[0]);
            if self.model.graph.inputs.contains(&node.inputs[0])
                || self.model.graph.inits.contains_key(&node.inputs[0])
                || self.propagated_inits.contains(&node.inputs[0])
            {
                self.propagated_inits.insert(node.outputs[0]);
            }
            let ty = get_c_type(inputs[0].elem_ty);
            created_calls.push(format!(
                "{} = ({ty} *){};",
                args[inputs.len()..][0],
                args[0]
            ))
        } else {
            let start_profiling = if self.enable_profiling {
                indent_all_by(4, "const struct timespec _start = now();".to_string())
            } else {
                String::new()
            };
            let end_profiling = if self.enable_profiling {
                indent_all_by(
                    4,
                    format!(
                        "const struct timespec _end = now();
const double start_in_sec = (double)_start.tv_sec + (double)_start.tv_nsec / 1e9;
const double end_in_sec = (double)_end.tv_sec + (double)_end.tv_nsec / 1e9;
elapsed_{opname} += end_in_sec - start_in_sec;",
                        opname = op.name()
                    ),
                )
            } else {
                String::new()
            };
            created_calls.push(format!(
                "{{
{start_profiling}
    {node_name}({});
{end_profiling}
}}",
                args.join(", ")
            ));
        }

        let kernel = match op {
            Op::Identity => self.translate_identity(&args, &inputs, outputs)?,
            Op::Conv2d(ref c) => self.translate_conv2d(c, &args, &inputs, outputs)?,
            Op::HardSigmoid(ref h) => self.translate_hard_sigmoid(h, &args, &inputs, outputs)?,
            Op::Add => self.translate_bin_op("+", &args, &inputs, outputs)?,
            Op::Sub => self.translate_bin_op("-", &args, &inputs, outputs)?,
            Op::Mul => self.translate_bin_op("*", &args, &inputs, outputs)?,
            Op::Div => self.translate_bin_op("/", &args, &inputs, outputs)?,
            Op::Greater => self.translate_bin_op(">", &args, &inputs, outputs)?,
            Op::Pow => self.translate_pow(node, &args, &inputs, outputs)?,
            Op::Sqrt => self.translate_sqrt(&args, &inputs, outputs)?,
            Op::ReLU => self.translate_relu(&args, &inputs, outputs)?,
            Op::Erf => self.translate_erf(&args, &inputs, outputs)?,
            Op::Sigmoid => self.translate_sigmoid(&args, &inputs, outputs)?,
            Op::Tanh => self.translate_tanh(&args, &inputs, outputs)?,
            Op::Where => self.translate_where(&args, &inputs, outputs)?,
            Op::GlobalAveragePool => self.translate_gavg_pool(&args, &inputs, outputs)?,
            Op::MaxPool(ref m) => self.translate_max_pool(m, &args, &inputs, outputs)?,
            Op::Reshape => self.translate_reshape(&args, &inputs, outputs)?,
            Op::MatMul => self.translate_mat_mul(&args, &inputs, outputs)?,
            Op::Flatten(ref f) => self.translate_flatten(f, &args, &inputs, outputs)?,
            Op::Gemm(ref g) => self.translate_gemm(g, &args, &inputs, outputs)?,
            Op::Transpose(ref t) => {
                self.translate_transpose(node_name.as_str(), t, &args, &inputs, outputs)?
            }
            Op::Expand => self.translate_expand(&args, &inputs, outputs)?,
            Op::Concat(ref c) => self.translate_concat(c, &args, &inputs, outputs)?,
            Op::Gather(ref g) => self.translate_gather(g, &args, &inputs, outputs)?,
            Op::ReduceMean(ref r) => self.translate_reduce_mean(r, &args, &inputs, outputs)?,
            Op::ReduceMax(ref r) => self.translate_reduce_max(r, &args, &inputs, outputs)?,
            Op::Softmax(ref s) => self.translate_softmax(s, &args, &inputs, outputs)?,
            Op::BatchNormalization(ref b) => {
                self.translate_batch_norm(b, &args, &inputs, outputs)?
            }
            Op::LayerNormalization(l) => self.translate_layer_norm(l, &args, &inputs, outputs)?,
            Op::Gelu => self.translate_gelu(&args, &inputs, outputs)?,
            Op::Unsqueeze(_) => String::new(), // nop
            Op::Squeeze(_) => String::new(),   // nop
            Op::Split(ref s) => self.translate_split(s, &args, &inputs, outputs)?,
            Op::Slice => self.translate_slice(&args, &inputs, outputs)?,
            Op::Cast(ref c) => self.translate_cast(c, &args, &inputs, outputs)?,
            Op::Resize(ref r) => self.translate_resize(r, &args, &inputs, outputs)?,
            Op::FusedElemwise(ref f) => {
                self.translate_fused_elemwise(f, &args, &inputs, outputs)?
            }
            _ => todo!("Translation not implemented for {:?}", op),
        };

        let decl = format!(
            "void {node_name}({args})",
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
                .join(", ")
        );
        self.created_kernel_protos.push(format!("{decl};"));

        if kernel == "/* Cranelift */" {
            let id = self.clif_ctx.module.declare_function(
                &node_name,
                Linkage::Export,
                &self.clif_ctx.ctx.func.signature,
            )?;
            self.clif_ctx
                .module
                .define_function(id, &mut self.clif_ctx.ctx)?;
            self.clif_ctx.module.clear_context(&mut self.clif_ctx.ctx);
        } else {
            let kernel = format!(
                "{decl} {{
{body}
}}",
                body = indent_all_by(4, kernel),
            );
            self.created_kernels.push(kernel);
        }

        Ok(())
    }

    fn translate_identity(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input = &inputs[0];
        let output = &outputs[0];

        assert_eq!(args.len(), 2);
        let ty = get_c_type(input.elem_ty);
        let input_name = &args[0];
        let output_name = &args[1];
        let size = output.dims.total_elems();

        Ok(format!(
            "memcpy({output_name}, {input_name}, sizeof({ty}) * {size});",
        ))
    }

    fn translate_conv2d(
        &mut self,
        op: &Conv2d,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];

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
            let num_threads = self.intra_op_num_threads;

            format!("float *col =
    (float *)malloc(sizeof(float) * {batch_size} * {input_c} * {output_h} * {output_w} * {kernel_h} * {kernel_w});

{{
    const int output_hw = {output_h} * {output_w};
    float *_input_ptr = (float *){input_name};
    float *_col_ptr = (float *)col;

    #pragma omp parallel for num_threads({num_threads})
    for (int outer = 0; outer < {batch_size} * {input_c}; outer++) {{
        float *input_ptr = _input_ptr + outer * {input_hw};
        float *col_ptr = _col_ptr + outer * {output_hw} * {kernel_h} * {kernel_w};
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

        let activation = op.activation.as_ref().map_or("".to_string(), |act| {
            let num_threads = self.intra_op_num_threads;
            let activation = match act {
                FusedActivation::Relu => "fmaxf(output_ptr[i], 0.0f)".to_string(),
                FusedActivation::HardSigmoid(HardSigmoid { alpha, beta }) => {
                    format!("fmaxf(0.0f, fminf(1.0f, output_ptr[i] * {alpha} + {beta}))")
                }
            };
            let size = output.dims.total_elems();
            indent_all_by(
                4,
                format!(
                    "{{
    float *output_ptr = (float *){output_name};
    #pragma omp parallel for num_threads({num_threads})
    #pragma clang loop vectorize(enable)
    for (int i = 0; i < {size}; i++) {{
        output_ptr[i] = {activation};
    }}
}}"
                ),
            )
        });

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

    {activation}
}}"
        );

        let kernel = format!(
            "{code_fill_bias}

{code_im2col}

{code_gemm}

free(col);"
        );

        Ok(kernel)
    }

    fn translate_hard_sigmoid(
        &mut self,
        hs: &HardSigmoid,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let alpha = hs.alpha;
        let beta = hs.beta;
        let size = inputs[0].dims.total_elems();
        let num_threads = self.intra_op_num_threads;
        let kernel = format!(
            "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = fminf(1.0, fmaxf(0.0, x * {alpha} + {beta}));
}}"
        );
        Ok(kernel)
    }

    fn translate_fused_elemwise(
        &mut self,
        op: &FusedElemwise,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        let kernel = if inputs[0].dims == inputs[1].dims {
            for inputs in inputs.windows(2) {
                assert!(inputs[0].elem_ty.is_f32());
                assert!(inputs[1].elem_ty.is_f32());
                assert_eq!(inputs[0].dims, inputs[1].dims);
            }

            let (expr, _) = op.chain.iter().fold(
                (String::new(), None),
                |(expr, last_expr_id), (op, op_ins, op_outs)| {
                    let mut args = vec![];
                    for op_in in op_ins {
                        if matches!(last_expr_id, Some(last) if *op_in == last) {
                            args.push(expr.clone());
                            continue;
                        }
                        args.push(format!("{}[i]", value_name(self.model, *op_in)));
                    }
                    let expr = match op {
                        Op::Add => format!("({} + {})", args[0], args[1]),
                        Op::Sub => format!("({} - {})", args[0], args[1]),
                        Op::Mul => format!("({} * {})", args[0], args[1]),
                        Op::Div => format!("({} / {})", args[0], args[1]),
                        Op::Sqrt => format!("sqrtf({})", args[0]),
                        Op::ReLU => format!("fmaxf({}, 0.0f)", args[0]),
                        _ => todo!("{:?} is not yet supported", op),
                    };
                    assert_eq!(op_outs.len(), 1);
                    (expr, Some(op_outs[0]))
                },
            );

            format!(
                "#pragma omp parallel for num_threads({th})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = {expr};
}}",
                th = self.intra_op_num_threads,
                size = outputs[0].dims.total_elems(),
            )
        } else {
            let rank = outputs[0].dims.len();
            let input_stride_list = inputs
                .iter()
                .map(|i| {
                    i.dims
                        .strides_for_broadcasting_to(&outputs[0].dims)
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let mut kernel = String::new();
            for (i, odim) in outputs[0].dims.iter().enumerate().rev() {
                if i == rank - 1 && rank > 1 {
                    kernel = format!(
                        "#pragma clang loop vectorize(enable)
for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    *output_ptr = {expr};
    {inc}
    output_ptr += 1;
}}",
                        expr = {
                            let mut out = format!("*input_0_ptr_{i}");
                            let mut opr_idx = 1;
                            for (op, inputs, _) in op.chain.iter() {
                                out = match op {
                                    Op::Add => format!("({out} + *input_{opr_idx}_ptr_{i})"),
                                    Op::Sub => format!("({out} - *input_{opr_idx}_ptr_{i})"),
                                    Op::Mul => format!("({out} * *input_{opr_idx}_ptr_{i})"),
                                    Op::Div => format!("({out} / *input_{opr_idx}_ptr_{i})"),
                                    Op::Pow => match self.model.graph.inits.get(&inputs[1]) {
                                        Some(init) if init.data::<f32>()[0] == 2. => {
                                            format!("({out} * {out})")
                                        }
                                        Some(init) if init.data::<f32>()[0] == 3. => {
                                            format!("({out} * {out} * {out})")
                                        }
                                        _ => format!("powf({out}, *input_{opr_idx}_ptr_{i})"),
                                    },
                                    Op::Sqrt => format!("sqrtf({out})"),
                                    _ => todo!("{op:?}"),
                                };
                                opr_idx += inputs.len() - 1;
                            }
                            out
                        },
                        inc = inputs
                            .iter()
                            .zip(input_names.iter())
                            .enumerate()
                            .map(|(k, (_shape, _name))| {
                                format!(
                                    "input_{k}_ptr_{i} += {step};",
                                    step = input_stride_list[k][i]
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                } else {
                    kernel = format!(
                        "{header}for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    {in_defs}
{body}
    {inc}
}}",
                        header = if i == 0 {
                            format!(
                                "{in_defs}
{out_ty} *output_ptr = {output_name};\n",
                                in_defs = inputs
                                    .iter()
                                    .zip(input_names.iter())
                                    .enumerate()
                                    .map(|(i, (shape, name))| {
                                        format!(
                                            "{} *input_{}_ptr_0 = ({} *){};",
                                            get_c_type(shape.elem_ty),
                                            i,
                                            get_c_type(shape.elem_ty),
                                            name
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                out_ty = get_c_type(outputs[0].elem_ty),
                            )
                        } else {
                            "".to_string()
                        },
                        in_defs = inputs
                            .iter()
                            .enumerate()
                            .map(|(k, shape)| {
                                format!(
                                    "{in_ty} *input_{k}_ptr_{iplus1} = input_{k}_ptr_{i};",
                                    in_ty = get_c_type(shape.elem_ty),
                                    iplus1 = i + 1
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                        body = indent_all_by(4, kernel),
                        inc = (0..inputs.len())
                            .map(|k| format!(
                                "input_{k}_ptr_{i} += {step};",
                                step = input_stride_list[k][i]
                            ))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                }
            }

            kernel
        };

        Ok(kernel)
    }

    fn translate_bin_op(
        &mut self,
        op: &str,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        let kernel = if inputs[0].dims == inputs[1].dims {
            format!(
                "#pragma omp parallel for num_threads({th})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output}[i] = {input_0}[i] {op} {input_1}[i];
}}",
                th = self.intra_op_num_threads,
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
            for (i, (((odim, ostr), i0str), i1str)) in outputs[0]
                .dims
                .iter()
                .zip(outputs[0].dims.strides().iter())
                .zip(input_0_strides.iter())
                .zip(input_1_strides.iter())
                .enumerate()
                .rev()
            {
                if i == rank - 1 && rank > 1 {
                    kernel = format!(
                        "#pragma clang loop vectorize(enable)
for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    *(output_ptr_{i} + {ostr} * i{i}) = 
        *(input_0_ptr_{i} + {i0str} * i{i}) {op}
        *(input_1_ptr_{i} + {i1str} * i{i});
}}"
                    );
                } else {
                    kernel = format!(
                        "{}
{omp}
for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    {in_ty} *input_0_ptr_{iplus1} = input_0_ptr_{i} + {i0str} * i{i};
    {in_ty} *input_1_ptr_{iplus1} = input_1_ptr_{i} + {i1str} * i{i};
    {out_ty} *output_ptr_{iplus1} = output_ptr_{i} + {ostr} * i{i};
{}
}}",
                        if i == 0 {
                            format!(
                                "{in_ty} *input_0_ptr_0 = ({in_ty} *){};
{in_ty} *input_1_ptr_0 = ({in_ty} *){};
{out_ty} *output_ptr_0 = {};\n",
                                input_names[0],
                                input_names[1],
                                output_name,
                                in_ty = get_c_type(inputs[0].elem_ty),
                                out_ty = get_c_type(outputs[0].elem_ty),
                            )
                        } else {
                            "".to_string()
                        },
                        indent_all_by(4, kernel),
                        iplus1 = i + 1,
                        in_ty = get_c_type(inputs[0].elem_ty),
                        out_ty = get_c_type(outputs[0].elem_ty),
                        omp = "// TODO",
                        // omp = if *odim == 12 {
                        //     format!("#pragma omp parallel for num_threads({th})")
                        // } else {
                        //     String::new()
                        // }
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
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        let num_threads = self.intra_op_num_threads;
        let kernel = if inputs[0].dims == inputs[1].dims {
            format!(
                "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = powf({input_0}[i], {input_1}[i]);
}}",
                input_0 = input_names[0],
                input_1 = input_names[1],
                size = outputs[0].dims.total_elems(),
            )
        } else if inputs[1].dims.is_scalar() {
            match self.model.graph.inits.get(&node.inputs[1]) {
                Some(init) if init.data::<f32>()[0] == 2. => format!(
                    "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = {input_0}[i] * {input_0}[i];
}}",
                    input_0 = input_names[0],
                    size = outputs[0].dims.total_elems(),
                ),
                Some(init) if init.data::<f32>()[0] == 3. => format!(
                    "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = {input_0}[i] * {input_0}[i] * {input_0}[i];
}}",
                    input_0 = input_names[0],
                    size = outputs[0].dims.total_elems(),
                ),
                _ => format!(
                    "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
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
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let size = inputs[0].dims.total_elems();
        let num_threads = self.intra_op_num_threads;
        let kernel = format!(
            "#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = fmaxf(0.0, x);
}}"
        );
        Ok(kernel)
    }

    fn translate_erf(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let size = inputs[0].dims.total_elems();
        let num_threads = self.intra_op_num_threads;
        let kernel = format!(
            "#pragma omp parallel for num_threads({num_threads})
for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = erff(x);
}}"
        );
        Ok(kernel)
    }

    fn translate_sigmoid(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let size = inputs[0].dims.total_elems();
        let num_threads = self.intra_op_num_threads;
        let kernel = format!(
            "
const float LOWER_RANGE        = -88.37626;
const float ROUNDING_BIAS      = 12582912.0;
const float LOG2RECIPROCAL     = 1.44269504088896341;
const float LOG2HIGH           = -6.93145752e-1;
const float LOG2LOW            = -1.42860677e-6;
const float POLY_0             = 0.0013780593872;
const float POLY_1             = 0.0083731245250;
const float POLY_2             = 0.0416695363820;
const float POLY_3             = 0.1666647195816;
const float POLY_4             = 0.4999998509884;
const float POLY_56            = 1.0000000000000;
const int32_t MAXIMUM_EXPONENT = 0x3F800000;

#pragma omp parallel for num_threads({num_threads})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    const float val0 = fmaxf(-{input_name}[i], LOWER_RANGE);
    const float biased = fmaf(val0, LOG2RECIPROCAL, ROUNDING_BIAS);
    const float m = biased - ROUNDING_BIAS;
    const float val1 = fmaf(m, LOG2HIGH, val0);
    const float val2 = fmaf(m, LOG2LOW, val1);
    const int32_t normal = (*(int *)&biased) << 23;
    const int32_t normal2 = normal + MAXIMUM_EXPONENT;
    const float p0 = POLY_0;
    const float p1 = fmaf(p0, val2, POLY_1);
    const float p2 = fmaf(p1, val2, POLY_2);
    const float p3 = fmaf(p2, val2, POLY_3);
    const float p4 = fmaf(p3, val2, POLY_4);
    const float p5 = fmaf(p4, val2, POLY_56);
    const float p6 = fmaf(p5, val2, POLY_56);
    const float p7 = p6 * (*(float *)&normal2);
    {output_name}[i] = 1.f / (1.f + p7);
}}
"
        );
        Ok(kernel)
    }

    fn translate_tanh(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[..inputs.len()][0];
        let output_name = &args[inputs.len()..][0];
        let size = inputs[0].dims.total_elems();
        let num_threads = self.intra_op_num_threads;
        let kernel = format!(
            "#pragma omp parallel for num_threads({num_threads})
for (int i = 0; i < {size}; i++) {{
    const float x = {input_name}[i];
    {output_name}[i] = tanhf(x);
}}"
        );
        Ok(kernel)
    }

    fn translate_where(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[..inputs.len()];
        let output_name = &args[inputs.len()..][0];

        // TODO: The following code is almost a copy of translate_bin_op.
        let kernel = if inputs[0].dims == inputs[1].dims && inputs[1].dims == inputs[2].dims {
            format!(
                "#pragma omp parallel for num_threads({th})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
    {output}[i] = {condition}[i] ? {input_0}[i] : {input_1}[i];
}}",
                th = self.intra_op_num_threads,
                condition = input_names[0],
                input_0 = input_names[1],
                input_1 = input_names[2],
                size = outputs[0].dims.total_elems(),
                output = output_name,
            )
        } else {
            let rank = outputs[0].dims.len();

            let condition_strides = inputs[0]
                .dims
                .strides_for_broadcasting_to(&outputs[0].dims)
                .unwrap();
            let input_0_strides = inputs[1]
                .dims
                .strides_for_broadcasting_to(&outputs[0].dims)
                .unwrap();
            let input_1_strides = inputs[2]
                .dims
                .strides_for_broadcasting_to(&outputs[0].dims)
                .unwrap();

            let mut kernel = String::new();
            for (i, (((odim, condstr), i0str), i1str)) in outputs[0]
                .dims
                .iter()
                .zip(condition_strides.iter())
                .zip(input_0_strides.iter())
                .zip(input_1_strides.iter())
                .enumerate()
                .rev()
            {
                if i == rank - 1 {
                    kernel = format!(
                        "// #pragma clang loop vectorize(enable) // TODO: Is this really impossible?
for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    *output_ptr = *cond_ptr_{i} ? *input_0_ptr_{i} : *input_1_ptr_{i};
    cond_ptr_{i} += {condstr};
    input_0_ptr_{i} += {i0str};
    input_1_ptr_{i} += {i1str};
    output_ptr += 1;
}}"
                    );
                } else {
                    kernel = format!(
                        "{}for (int i{i} = 0; i{i} < {odim}; i{i}++) {{
    unsigned char *cond_ptr_{iplus1} = cond_ptr_{i};
    float *input_0_ptr_{iplus1} = input_0_ptr_{i};
    float *input_1_ptr_{iplus1} = input_1_ptr_{i};
{}
    cond_ptr_{i} += {condstr};
    input_0_ptr_{i} += {i0str};
    input_1_ptr_{i} += {i1str};
}}",
                        if i == 0 {
                            format!(
                                "unsigned char *cond_ptr_0 = (unsigned char *){};
float *input_0_ptr_0 = (float *){};
float *input_1_ptr_0 = (float *){};
float *output_ptr = {};\n",
                                input_names[0], input_names[1], input_names[2], output_name
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

    fn translate_gavg_pool(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];

        let input = inputs[0];
        let output = &outputs[0];

        assert!(input.dims.len() == 4);
        assert!(output.dims.len() == 4);

        let Some(&[n, c, h, w]) = input.dims.get(0..4) else {
            return Err(SessionError::Message(
                "Input must be four dimensions".into(),
            ));
        };
        let Some(&[isn, isc, _, _]) = input.dims.strides().get(0..4) else {
            panic!()
        };
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
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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
        _args: &[String],
        _inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        // `Reshape`s are handled as nop.
        Ok(String::new())

        // Old implementation:
        // let input_name = &args[..inputs.len()][0];
        // let output_name = &args[inputs.len()..][0];
        // let ty = get_c_type(inputs[0].elem_ty);
        // let kernel = format!(
        //     "memcpy({output_name}, {input_name}, {size} * sizeof({ty}));",
        //     size = inputs[0].dims.total_elems()
        // );
        // Ok(kernel)
    }

    fn translate_mat_mul(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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
        } else if input_0.dims.len() == 3 && input_1.dims.len() == 3 {
            let [batch, m, _k] = input_0.dims.to_fixed_dims::<3>();
            let [batch_, k, n] = input_1.dims.to_fixed_dims::<3>();
            assert_eq!(batch, batch_);

            let kernel = format!(
                "for (int i = 0; i < {batch}; i++) {{
    const float *input_0_ptr = {input_0} + i * ({m} * {k});
    const float *input_1_ptr = {input_1} + i * ({k} * {n});
    float *output_ptr = {output} + i * ({m} * {n});
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        {m}, {n}, {k}, 1.,
        input_0_ptr, {k}, input_1_ptr, {n}, 0., output_ptr, {n});
}}",
                input_0 = input_names[0],
                input_1 = input_names[1],
                output = output_names[0],
            );

            Ok(kernel)
        } else if input_0.dims.len() == 4 && input_1.dims.len() == 4 {
            let [b0, b1, m, _k] = input_0.dims.to_fixed_dims::<4>();
            let [b0_, b1_, k, n] = input_1.dims.to_fixed_dims::<4>();
            assert_eq!(b0, b0_);
            assert_eq!(b1, b1_);
            let batch = b0 * b1;

            let input_0 = &input_names[0];
            let input_1 = &input_names[1];
            let output = &output_names[0];

            fn repeat<F: Fn(usize) -> String>(f: F, batch: usize) -> String {
                (0..batch).map(f).collect::<Vec<_>>().join(", ")
            }
            let input_0_ptrs_init = repeat(|i| format!("{input_0} + {i} * ({m} * {k})"), batch);
            let input_1_ptrs_init = repeat(|i| format!("{input_1} + {i} * ({k} * {n})"), batch);
            let output_ptrs_init = repeat(|i| format!("{output} + {i} * ({m} * {n})"), batch);
            let m_array_init = repeat(|_| format!("{m}"), batch);
            let n_array_init = repeat(|_| format!("{n}"), batch);
            let k_array_init = repeat(|_| format!("{k}"), batch);
            let alpha_array_init = repeat(|_| String::from("1.0f"), batch);
            let beta_array_init = repeat(|_| String::from("0.0f"), batch);
            let trans_a_init = repeat(|_| String::from("CblasNoTrans"), batch);
            let trans_b_init = repeat(|_| String::from("CblasNoTrans"), batch);
            let group_size_init = repeat(|_| String::from("1"), batch);
            let kernel = format!(
                "#ifdef BLIS_ENABLE_CBLAS
    const float *input_0_ptrs[{batch}] = {{ {input_0_ptrs_init} }};
    const float *input_1_ptrs[{batch}] = {{ {input_1_ptrs_init} }};
    float *output_ptrs[{batch}] = {{ {output_ptrs_init} }};
    int m_array[{batch}] = {{ {m_array_init} }};
    int n_array[{batch}] = {{ {n_array_init} }};
    int k_array[{batch}] = {{ {k_array_init} }};
    int lda_array[{batch}] = {{ {k_array_init} }};
    int ldb_array[{batch}] = {{ {n_array_init} }};
    int ldc_array[{batch}] = {{ {n_array_init} }};
    float alpha_array[{batch}] = {{ {alpha_array_init} }};
    float beta_array[{batch}] = {{ {beta_array_init} }};
    int group_size[{batch}] = {{ {group_size_init} }};
    enum CBLAS_TRANSPOSE trans_a[{batch}] = {{ {trans_a_init} }};
    enum CBLAS_TRANSPOSE trans_b[{batch}] = {{ {trans_b_init} }};

    cblas_sgemm_batch(CblasRowMajor,
        trans_a, trans_b,
        m_array, n_array, k_array,
        alpha_array, (const float **)input_0_ptrs, lda_array,
        (const float **)input_1_ptrs, ldb_array,
        beta_array, output_ptrs, ldc_array,
        {batch}, group_size);
#else
    for (size_t i = 0; i < {batch}; i++) {{
        const float *input_0_ptr = {input_0} + i * ({m} * {k});
        const float *input_1_ptr = {input_1} + i * ({k} * {n});
        float *output_ptr = {output} + i * ({m} * {n});
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            {m}, {n}, {k}, 1.,
            input_0_ptr, {k}, input_1_ptr, {n}, 0., output_ptr, {n});
    }}
#endif"
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
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
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
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = &outputs[0];
        let input_names = &args[..inputs.len()];
        let output_names = &args[inputs.len()..];

        assert!(input_0.dims.len() == 2);
        assert!(input_1.dims.len() == 2);
        assert!(output.elem_ty.is_f32());

        let m = input_0.dims[gemm.trans_a as usize];
        let k = input_0.dims[1 - gemm.trans_a as usize];
        let n = input_1.dims[1 - gemm.trans_b as usize];

        let bias = if let Some(input_2) = inputs.get(2) {
            match input_2.dims.len() {
                1 => format!(
                    "for (int i = 0; i < {output_size}; i += {n}) {{
    memcpy({out} + i, {in2}, {n} * sizeof(float));
}}",
                    output_size = output.dims.total_elems(),
                    in2 = input_names[2],
                    out = output_names[0]
                ),
                2 => {
                    assert_eq!(output.dims.total_elems(), m * n);
                    format!(
                        "memcpy({out}, {bias}, {m} * {n} * sizeof(float));",
                        bias = input_names[2],
                        out = output_names[0]
                    )
                }
                _ => unreachable!(),
            }
        } else {
            String::new()
        };
        let kernel = format!(
            "{bias}
cblas_sgemm(CblasRowMajor, {transa}, {transb},
    {m}, {n}, {k}, {alpha},
    {in0}, {lda}, {in1}, {ldb}, {beta}, {out}, {n});",
            bias = bias,
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
            in0 = input_names[0],
            in1 = input_names[1],
            lda = if gemm.trans_a { m } else { k },
            ldb = if gemm.trans_b { k } else { n },
            out = output_names[0],
            alpha = gemm.alpha,
            beta = inputs.get(2).map_or_else(|| 0.0, |_| gemm.beta)
        );

        Ok(kernel)
    }

    fn translate_transpose(
        &mut self,
        name: &str,
        transpose: &Transpose,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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

        if self.enable_clif {
            if num_blocks == 1 {
                return self.translate_transpose_clif(
                    num_elems_in_block,
                    num_blocks,
                    inputs[0].elem_ty,
                );
            } else {
                let indices = (0..num_blocks)
                    .scan(0, |src_idx, _| {
                        let idx = *src_idx;
                        *src_idx = next_index(&mut perm, *src_idx);
                        Some(idx as u64)
                    })
                    .collect::<Vec<_>>();
                if num_elems_in_block == 1 {
                    return self.translate_transpose_clif_2(
                        name,
                        indices,
                        num_elems_in_block,
                        num_blocks,
                        inputs[0].elem_ty,
                    );
                } else {
                    return self.translate_transpose_clif_3(
                        name,
                        indices,
                        num_elems_in_block,
                        num_blocks,
                        inputs[0].elem_ty,
                    );
                }
            }
        }

        let kernel = if num_blocks == 1 {
            format!("memcpy({output_name}, {input_name}, sizeof(float) * {num_elems_in_block});")
        } else {
            let indices = (0..num_blocks)
                .scan(0, |src_idx, _| {
                    let idx = *src_idx;
                    *src_idx = next_index(&mut perm, *src_idx);
                    Some(idx.to_string())
                })
                .collect::<Vec<_>>()
                .join(", ");
            if num_elems_in_block == 1 {
                format!(
                    "int src_indices[{num_blocks}] = {{ {indices} }};
for (int i = 0; i < {num_blocks}; i++) {{
    {output_name}[i] = {input_name}[src_indices[i]];
}}",
                )
            } else {
                format!(
                    "int src_indices[{num_blocks}] = {{ {indices} }};
for (int i = 0; i < {num_blocks}; i++) {{
    memcpy({output_name} + i * {num_elems_in_block},
            {input_name} + src_indices[i],
            sizeof(float) * {num_elems_in_block});
}}",
                )
            }
        };

        Ok(kernel)
    }

    fn translate_transpose_clif(
        &mut self,
        num_elems_in_block: usize,
        num_blocks: usize,
        elem_ty: TensorElemType,
    ) -> Result<String, SessionError> {
        assert_eq!(num_blocks, 1); // TODO

        let mut func = Function::new();
        let mut builder_ctx = FunctionBuilderContext::new();

        let ptr = self.clif_ctx.module.target_config().pointer_type();
        for _param in 0..2 {
            func.signature.params.push(ir::AbiParam::new(ptr));
        }

        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
        let entry = builder.create_block();

        {
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let elem_sz = get_clif_type(elem_ty).lane_bits() / 8;
            let input_param = builder.block_params(entry)[0];
            let output_param = builder.block_params(entry)[1];
            let size = builder
                .ins()
                .iconst(ptr, elem_sz as i64 * num_elems_in_block as i64);
            let dst = output_param;
            let src = input_param;
            builder.call_memcpy(self.clif_ctx.module.target_config(), dst, src, size);
            builder.ins().return_(&[]);
        }

        builder.seal_block(entry);
        builder.finalize();

        self.clif_ctx.ctx.func = func;

        Ok("/* Cranelift */".to_string())
    }

    fn translate_transpose_clif_2(
        &mut self,
        name: &str,
        indices: Vec<u64>,
        num_elems_in_block: usize,
        num_blocks: usize,
        elem_ty: TensorElemType,
    ) -> Result<String, SessionError> {
        assert_eq!(num_elems_in_block, 1);

        let mut func = Function::new();
        let mut builder_ctx = FunctionBuilderContext::new();

        let ptr = self.clif_ctx.module.target_config().pointer_type();
        for _param in 0..2 {
            func.signature.params.push(ir::AbiParam::new(ptr));
        }

        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
        let entry = builder.create_block();
        let body = builder.create_block();
        let merge = builder.create_block();

        let indices_id = self.clif_ctx.module.declare_data(
            format!("{name}.indices.{}", indices.len()).as_str(),
            Linkage::Local,
            false,
            false,
        )?;
        let mut desc = DataDescription::new();
        desc.define(
            unsafe {
                let mut indices = ManuallyDrop::new(indices);
                Vec::from_raw_parts(
                    indices.as_mut_ptr() as *mut u8,
                    indices.len() * size_of::<usize>(),
                    indices.capacity() * size_of::<usize>(),
                )
            }
            .into_boxed_slice(),
        );
        self.clif_ctx.module.define_data(indices_id, &desc)?;
        let gbl_indices = self
            .clif_ctx
            .module
            .declare_data_in_func(indices_id, builder.func);

        // int src_indices[{num_blocks}] = { {indices} };
        // for (int i = 0; i < {num_blocks}; i++) {
        //     {out}[i] = {in}[src_indices[i]];
        // }
        let var_input;
        let var_output;
        let var_counter;
        let var_indices;
        let zero;
        {
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let input_param = builder.block_params(entry)[0];
            let output_param = builder.block_params(entry)[1];
            zero = builder.ins().iconst(I64, 0);
            var_input = self.clif_ctx.create_var(ptr, input_param, &mut builder);
            var_output = self.clif_ctx.create_var(ptr, output_param, &mut builder);
            let max = builder.ins().iconst(I64, num_blocks as i64);
            var_counter = self.clif_ctx.create_var(I64, max, &mut builder);
            let val_indices = builder.ins().global_value(I64, gbl_indices);
            var_indices = self.clif_ctx.create_var(ptr, val_indices, &mut builder);
            builder.ins().jump(body, &[]);
        }
        {
            builder.switch_to_block(body);
            let elem_ty = get_clif_type(elem_ty);
            let elem_sz = elem_ty.lane_bits() / 8;

            // Changing the instruction order here might cause a performance degradation.
            let val_indices = builder.use_var(var_indices);
            let in_idx = builder.ins().load(I64, MemFlags::new(), val_indices, 0);
            let next_indices = builder.ins().iadd_imm(val_indices, 8);
            builder.def_var(var_indices, next_indices);
            assert!(elem_sz.is_power_of_two());
            let off = builder.ins().ishl_imm(in_idx, elem_sz.ilog2() as i64);
            let val_input = builder.use_var(var_input);
            let ptr_input = builder.ins().iadd(val_input, off);
            let val_in = builder.ins().load(elem_ty, MemFlags::new(), ptr_input, 0);
            let val_output = builder.use_var(var_output);
            builder.ins().store(MemFlags::new(), val_in, val_output, 0);
            let next_output = builder.ins().iadd_imm(val_output, elem_sz as i64);
            builder.def_var(var_output, next_output);

            let val_counter = builder.use_var(var_counter);
            let dec_counter = builder.ins().iadd_imm(val_counter, -1);
            builder.def_var(var_counter, dec_counter);

            let cond = builder.ins().icmp(IntCC::NotEqual, zero, dec_counter);
            builder.ins().brif(cond, body, &[], merge, &[]);
        }
        {
            builder.switch_to_block(merge);
            builder.ins().return_(&[]);
        }

        builder.seal_block(entry);
        builder.seal_block(body);
        builder.seal_block(merge);
        builder.finalize();

        self.clif_ctx.ctx.func = func;

        Ok("/* Cranelift */".to_string())
    }

    fn translate_transpose_clif_3(
        &mut self,
        name: &str,
        indices: Vec<u64>,
        num_elems_in_block: usize,
        num_blocks: usize,
        elem_ty: TensorElemType,
    ) -> Result<String, SessionError> {
        let mut func = Function::new();
        let mut builder_ctx = FunctionBuilderContext::new();

        let ptr = self.clif_ctx.module.target_config().pointer_type();
        for _param in 0..2 {
            func.signature.params.push(ir::AbiParam::new(ptr));
        }

        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
        let entry = builder.create_block();
        let body = builder.create_block();
        let merge = builder.create_block();

        let indices_id = self.clif_ctx.module.declare_data(
            format!("{name}.indices.{}", indices.len()).as_str(),
            Linkage::Local,
            false,
            false,
        )?;
        let mut desc = DataDescription::new();
        desc.define(
            unsafe {
                let mut indices = ManuallyDrop::new(indices);
                Vec::from_raw_parts(
                    indices.as_mut_ptr() as *mut u8,
                    indices.len() * size_of::<usize>(),
                    indices.capacity() * size_of::<usize>(),
                )
            }
            .into_boxed_slice(),
        );
        self.clif_ctx.module.define_data(indices_id, &desc)?;
        let gbl_indices = self
            .clif_ctx
            .module
            .declare_data_in_func(indices_id, builder.func);

        // int src_indices[{num_blocks}] = { {indices} };
        // for (int i = 0; i < {num_blocks}; i++) {
        //     memcpy({out} + i * {num_elems_in_block},
        //            {in} + src_indices[i],
        //            sizeof(float) * {num_elems_in_block});
        // }
        let var_input;
        let var_output;
        let var_counter;
        let var_indices;
        let zero;
        {
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let input_param = builder.block_params(entry)[0];
            let output_param = builder.block_params(entry)[1];
            zero = builder.ins().iconst(I64, 0);
            var_input = self.clif_ctx.create_var(ptr, input_param, &mut builder);
            var_output = self.clif_ctx.create_var(ptr, output_param, &mut builder);
            let max = builder.ins().iconst(I64, num_blocks as i64);
            var_counter = self.clif_ctx.create_var(I64, max, &mut builder);
            let val_indices = builder.ins().global_value(I64, gbl_indices);
            var_indices = self.clif_ctx.create_var(ptr, val_indices, &mut builder);
            builder.ins().jump(body, &[]);
        }
        {
            builder.switch_to_block(body);
            let elem_ty = get_clif_type(elem_ty);
            let elem_sz = elem_ty.lane_bits() / 8;

            // Changing the instruction order here might cause a performance degradation.

            let val_indices = builder.use_var(var_indices);
            let in_idx = builder.ins().load(I64, MemFlags::new(), val_indices, 0);
            let next_indices = builder.ins().iadd_imm(val_indices, 8);
            builder.def_var(var_indices, next_indices);
            assert!(elem_sz.is_power_of_two());

            let off = builder.ins().ishl_imm(in_idx, elem_sz.ilog2() as i64);
            let val_input = builder.use_var(var_input);
            let ptr_input = builder.ins().iadd(val_input, off);
            let src = ptr_input;

            let val_output = builder.use_var(var_output);
            let next_output = builder
                .ins()
                .iadd_imm(val_output, elem_sz as i64 * num_elems_in_block as i64);
            builder.def_var(var_output, next_output);
            let dest = val_output;

            let size = builder
                .ins()
                .iconst(I64, elem_sz as i64 * num_elems_in_block as i64);

            builder.call_memcpy(self.clif_ctx.module.target_config(), dest, src, size);

            let val_counter = builder.use_var(var_counter);
            let dec_counter = builder.ins().iadd_imm(val_counter, -1);
            builder.def_var(var_counter, dec_counter);

            let cond = builder.ins().icmp(IntCC::NotEqual, zero, dec_counter);
            builder.ins().brif(cond, body, &[], merge, &[]);
        }
        {
            builder.switch_to_block(merge);
            builder.ins().return_(&[]);
        }

        builder.seal_block(entry);
        builder.seal_block(body);
        builder.seal_block(merge);
        builder.finalize();

        self.clif_ctx.ctx.func = func;

        Ok("/* Cranelift */".to_string())
    }

    fn translate_expand(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input = inputs[0];
        let output = &outputs[0];
        let input_name = &args[0];
        let output_name = &args[inputs.len()];

        assert!(input.dims.len() == 4);
        assert!(input.dims[0..3] == [1, 1, 1]);
        assert!(output.dims.len() == 4);
        assert!(input.elem_ty.is_i64());

        let kernel = format!(
            "for (int i = 0; i < {out_size}; i++) {{
    {output_name}[i] = {input_name}[i % {in_size}];
}}",
            out_size = output.dims.total_elems(),
            in_size = input.dims.total_elems(),
        );

        Ok(kernel)
    }

    fn translate_concat(
        &mut self,
        concat: &Concat,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        if self.enable_clif {
            return self.translate_concat_clif(concat, args, inputs, outputs);
        }

        let input_names = &args[0..inputs.len()];
        let output_name = &args[inputs.len()];
        let output = &outputs[0];

        assert!(output.elem_ty.is_f32());
        let axis = if concat.axis < 0 {
            (output.dims.len() as i64 + concat.axis) as usize
        } else {
            concat.axis as usize
        };
        assert!(axis < output.dims.len());

        let sum_num_elems = inputs
            .iter()
            .map(|input| {
                assert_eq!(input.dims.len(), output.dims.len());
                assert_eq!(input.dims[0..axis], output.dims[0..axis]);
                assert_eq!(input.dims[axis + 1..], output.dims[axis + 1..]);
                input.dims[axis..].iter().product::<usize>()
            })
            .sum::<usize>();
        let outer = output.dims.total_elems() / sum_num_elems;

        let kernel = format!(
            "int offset = 0;
for (int i = 0; i < {outer}; i++) {{
{memcpy}
}}",
            memcpy = indent_all_by(4, input_names.iter().zip(inputs.iter()).map(|(name, input)| {
                let num_elems = input.dims[axis..].iter().product::<usize>();
                let ty = get_c_type(output.elem_ty);
                format!("memcpy({output_name} + offset, {name} + i * {num_elems}, sizeof({ty}) * {num_elems});\n\
                         offset += {num_elems};")
            }).collect::<Vec<_>>().join("\n"))
        );

        Ok(kernel)
    }

    fn translate_concat_clif(
        &mut self,
        concat: &Concat,
        _args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let output = &outputs[0];
        assert!(output.elem_ty.is_f32());

        let axis = if concat.axis < 0 {
            (output.dims.len() as i64 + concat.axis) as usize
        } else {
            concat.axis as usize
        };
        assert!(axis < output.dims.len());

        let outer;
        {
            let total_elems_along_axis = inputs
                .iter()
                .map(|input| {
                    assert_eq!(input.dims.len(), output.dims.len());
                    assert_eq!(input.dims[0..axis], output.dims[0..axis]);
                    assert_eq!(input.dims[axis + 1..], output.dims[axis + 1..]);
                    input.dims[axis..].iter().product::<usize>()
                })
                .sum::<usize>();
            outer = output.dims.total_elems() / total_elems_along_axis;
        }

        let mut func = Function::new();
        let mut builder_ctx = FunctionBuilderContext::new();

        let ptr = self.clif_ctx.module.target_config().pointer_type();
        for _param in 0..inputs.len() + outputs.len() {
            func.signature.params.push(ir::AbiParam::new(ptr));
        }

        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
        let entry = builder.create_block();
        let header = builder.create_block();
        let body = builder.create_block();
        let merge = builder.create_block();

        // int offset = 0;
        // for (int i = 0; i < {outer}; i++) {
        //     memcpy({output_name} + offset, {name} + i * {num_elems}, sizeof({ty}) * {num_elems});
        //     offset += {num_elems};
        // }
        let counter;
        let offset;
        let input_params;
        let output_param;
        {
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let zero = builder.ins().iconst(I64, 0);
            #[allow(clippy::unnecessary_to_owned)]
            let params = builder.block_params(entry)[..inputs.len()].to_vec();
            input_params = params
                .into_iter()
                .map(|i| self.clif_ctx.create_var(ptr, i, &mut builder))
                .collect::<Vec<_>>();
            output_param = self.clif_ctx.create_var(
                ptr,
                builder.block_params(entry)[inputs.len()],
                &mut builder,
            );
            counter = self.clif_ctx.create_var(I64, zero, &mut builder);
            offset = self.clif_ctx.create_var(I64, zero, &mut builder);
            builder.ins().jump(header, &[]);
        }
        {
            builder.switch_to_block(header);
            let max = builder.ins().iconst(I64, outer as i64);
            let counter_ = builder.use_var(counter);
            let cond = builder.ins().icmp(IntCC::UnsignedLessThan, counter_, max);
            builder.ins().brif(cond, body, &[], merge, &[]);
        }
        {
            builder.switch_to_block(body);
            let val_counter = builder.use_var(counter);
            for (param, shape) in input_params.iter().zip(inputs.iter()) {
                let val_offset = builder.use_var(offset);
                let num_elems = shape.dims[axis..].iter().product::<usize>();
                let tysz = get_clif_type(output.elem_ty).lane_bits() / 8;
                let size = builder.ins().iconst(I64, tysz as i64 * num_elems as i64);
                let inc = builder.ins().imul_imm(val_offset, tysz as i64);
                let out_ = builder.use_var(output_param);
                let dest = builder.ins().iadd(out_, inc);
                let inc = builder
                    .ins()
                    .imul_imm(val_counter, tysz as i64 * num_elems as i64);
                let val_input = builder.use_var(*param);
                let src = builder.ins().iadd(val_input, inc);
                builder.call_memcpy(self.clif_ctx.module.target_config(), dest, src, size);
                let new_offset = builder.ins().iadd_imm(val_offset, num_elems as i64);
                builder.def_var(offset, new_offset);
            }
            let inc_counter = builder.ins().iadd_imm(val_counter, 1);
            builder.def_var(counter, inc_counter);
            builder.ins().jump(header, &[]);
        }
        {
            builder.switch_to_block(merge);
            builder.ins().return_(&[]);
        }

        builder.seal_block(entry);
        builder.seal_block(header);
        builder.seal_block(body);
        builder.seal_block(merge);

        builder.finalize();

        self.clif_ctx.ctx.func = func;

        Ok("/* Cranelift */".to_string())
    }

    fn translate_gather(
        &mut self,
        gather: &Gather,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        if self.enable_clif {
            return self.translate_gather_clif(gather, args, inputs, outputs);
        }

        let data = inputs[0];
        let indices = inputs[1];
        let _output = &outputs[0];
        let data_name = &args[0];
        let indices_name = &args[1];
        let output_name = &args[inputs.len()..][0];

        assert!(data.elem_ty.is_f32());
        assert!(gather.axis >= 0);
        assert!(
            indices.dims.is_scalar() || (indices.dims.len() == 2 && indices.dims[0] == 1),
            "Unsupported indices shape: {:?}",
            indices.dims
        );

        if indices.dims.is_scalar() {
            let axis = gather.axis as usize;
            data.dims[..axis].iter().for_each(|&d| assert_eq!(d, 1));

            let kernel = format!(
                "memcpy({out}, {data} + {stride} * ({ind}[0]), sizeof(float) * {stride});",
                out = output_name,
                data = data_name,
                stride = data.dims.strides()[axis],
                ind = indices_name,
            );
            Ok(kernel)
        } else {
            let axis = gather.axis as usize;
            assert_eq!(axis, 0);

            let len = indices.dims.total_elems();
            let size = data.dims[1];
            let stride = data.dims.strides()[axis];
            let kernel = format!(
                "for (int i = 0; i < {len}; i++) {{
    memcpy({output_name} + i * {size}, {data_name} + {stride} * ({indices_name}[i]), sizeof(float) * {size});
}}"
            );
            Ok(kernel)
        }
    }

    fn translate_gather_clif(
        &mut self,
        gather: &Gather,
        _args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let data = inputs[0];
        let indices = inputs[1];
        let _output = &outputs[0];
        let mut func = Function::new();

        assert!(data.elem_ty.is_f32());
        assert!(gather.axis >= 0);
        assert!(
            indices.dims.is_scalar() || (indices.dims.len() == 2 && indices.dims[0] == 1),
            "Unsupported indices shape: {:?}",
            indices.dims
        );
        assert!(indices.elem_ty.is_i64());

        if indices.dims.is_scalar() {
            let axis = gather.axis as usize;
            assert_eq!(axis, 1);
            assert_eq!(data.dims.len(), 3);
            assert_eq!(data.dims[0], 1);

            let mut builder_ctx = FunctionBuilderContext::new();

            let ptr = self.clif_ctx.module.target_config().pointer_type();
            for _param in 0..inputs.len() + outputs.len() {
                func.signature.params.push(AbiParam::new(ptr))
            }

            let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
            let entry = builder.create_block();

            // memcpy({output}, {data} + data.dims.strides()[axis]} * ({indices}[0]), sizeof(float) * {data.dims[2]});
            {
                builder.switch_to_block(entry);
                builder.append_block_params_for_function_params(entry);
                #[allow(clippy::unnecessary_to_owned)]
                let params = builder.block_params(entry)[..inputs.len()].to_vec();
                let [var_data, var_indices] = params
                    .into_iter()
                    .map(|i| self.clif_ctx.create_var(ptr, i, &mut builder))
                    .collect::<Vec<_>>()[..]
                else {
                    panic!()
                };
                let var_out = self.clif_ctx.create_var(
                    ptr,
                    builder.block_params(entry)[inputs.len()],
                    &mut builder,
                );
                let tysz = get_clif_type(data.elem_ty).lane_bits() / 8;
                let off = builder
                    .ins()
                    .iconst(I64, tysz as i64 * data.dims.strides()[axis] as i64);
                let val_indices = builder.use_var(var_indices);
                let idx = builder.ins().load(I64, MemFlags::new(), val_indices, 0);
                let src = builder.ins().imul(idx, off);
                let val_data = builder.use_var(var_data);
                let src = builder.ins().iadd(val_data, src);
                let val_out = builder.use_var(var_out);
                let size = builder.ins().iconst(I64, tysz as i64 * data.dims[2] as i64);
                builder.call_memcpy(self.clif_ctx.module.target_config(), val_out, src, size);
                builder.ins().return_(&[]);
            }

            builder.seal_block(entry);
            builder.finalize();
        } else {
            let axis = gather.axis as usize;
            assert_eq!(axis, 0);

            let len = indices.dims.total_elems();
            let size = data.dims[1];
            let stride = data.dims.strides()[axis];

            // for (int i = 0; i < {len}; i++) {
            //     memcpy(
            //         {output_name} + i * {size},
            //         {data_name} + {stride} * ({indices_name}[i]),
            //         sizeof(float) * {size});
            // }

            let mut builder_ctx = FunctionBuilderContext::new();

            let ptr = self.clif_ctx.module.target_config().pointer_type();
            for _param in 0..inputs.len() + outputs.len() {
                func.signature.params.push(AbiParam::new(ptr))
            }

            let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
            let entry = builder.create_block();
            let header = builder.create_block();
            let body = builder.create_block();
            let merge = builder.create_block();

            let var_data;
            let var_indices;
            let var_out;
            let var_counter;
            {
                builder.switch_to_block(entry);
                builder.append_block_params_for_function_params(entry);
                #[allow(clippy::unnecessary_to_owned)]
                let var_inputs = builder.block_params(entry)[..inputs.len()]
                    .to_vec()
                    .into_iter()
                    .map(|i| self.clif_ctx.create_var(ptr, i, &mut builder))
                    .collect::<Vec<_>>();
                var_data = var_inputs[0];
                var_indices = var_inputs[1];
                var_out = self.clif_ctx.create_var(
                    ptr,
                    builder.block_params(entry)[inputs.len()],
                    &mut builder,
                );
                let zero = builder.ins().iconst(I64, 0);
                var_counter = self.clif_ctx.create_var(I64, zero, &mut builder);
                builder.ins().jump(header, &[]);
            }
            {
                builder.switch_to_block(header);
                let max = builder.ins().iconst(I64, len as i64);
                let counter = builder.use_var(var_counter);
                let cond = builder.ins().icmp(IntCC::UnsignedLessThan, counter, max);
                builder.ins().brif(cond, body, &[], merge, &[]);
            }
            {
                builder.switch_to_block(body);
                let counter = builder.use_var(var_counter);
                let tysz = get_clif_type(data.elem_ty).lane_bits() / 8;
                let off = builder.ins().iconst(I64, tysz as i64 * stride as i64);
                let indices = builder.use_var(var_indices);
                let t = builder.ins().imul_imm(counter, 8);
                let idx_addr = builder.ins().iadd(indices, t);
                let idx = builder.ins().load(I64, MemFlags::new(), idx_addr, 0);
                let src = builder.ins().imul(idx, off);
                let data = builder.use_var(var_data);
                let src = builder.ins().iadd(data, src);
                let val_out = builder.use_var(var_out);
                let idx = builder.ins().imul_imm(counter, tysz as i64 * size as i64);
                let dst = builder.ins().iadd(val_out, idx);
                let size = builder.ins().iconst(I64, tysz as i64 * size as i64);
                builder.call_memcpy(self.clif_ctx.module.target_config(), dst, src, size);
                let inc_counter = builder.ins().iadd_imm(counter, 1);
                builder.def_var(var_counter, inc_counter);
                builder.ins().jump(header, &[]);
            }
            {
                builder.switch_to_block(merge);
                builder.ins().return_(&[]);
            }

            builder.seal_block(entry);
            builder.seal_block(header);
            builder.seal_block(body);
            builder.seal_block(merge);
            builder.finalize();
        }

        self.clif_ctx.ctx.func = func;

        Ok("/* Cranelift */".to_string())
    }

    fn translate_reduce_mean(
        &mut self,
        rmean: &ReduceMean,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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
        assert!(
            axes.iter()
                .all(|&axis| axis >= input.dims.len() - axes.len()),
            "Axes must specify innermost dimensions"
        );
        assert!(rmean.keep_dims);

        let kernel = format!(
            "for (int i = 0; i < {batch}; i++) {{
    const float *input_ptr = {input_name} + i * {axis_len};
    float sum = 0.f;
    #pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        sum += input_ptr[j];
    }}
    {output_name}[i] = sum * (1.f / {axis_len});
}}",
            batch = output.dims.total_elems(),
            axis_len = input.dims[axes[0]..].iter().product::<usize>(),
        );

        Ok(kernel)
    }

    fn translate_reduce_max(
        &mut self,
        rmax: &ReduceMax,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        // TODO: Current implementation only supports reduction over all axes.

        let input = inputs[0];
        let input_name = &args[0];
        let output_name = &args[1];
        assert!(rmax.axes.is_empty());
        assert!(!rmax.keep_dims);

        let outer = input.dims.total_elems();
        let kernel = format!(
            "float max = -INFINITY;
#pragma clang loop vectorize(enable)
for (int i = 0; i < {outer}; i++) {{
    max = fmaxf({input_name}[i], max);
}}
{output_name}[0] = max;
"
        );

        Ok(kernel)
    }

    fn translate_softmax(
        &mut self,
        softmax: &Softmax,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input = inputs[0];
        let output = &outputs[0];
        let input_name = &args[0];
        let output_name = &args[1];

        assert!(softmax.axis == -1 || softmax.axis == (input.dims.len() - 1) as i64);

        let axis_len = *input.dims.last().unwrap();

        let kernel = format!(
            "
const float LOWER_RANGE        = -88.37626;
const float ROUNDING_BIAS      = 12582912.0;
const float LOG2RECIPROCAL     = 1.44269504088896341;
const float LOG2HIGH           = -6.93145752e-1;
const float LOG2LOW            = -1.42860677e-6;
const float POLY_0             = 0.0013780593872;
const float POLY_1             = 0.0083731245250;
const float POLY_2             = 0.0416695363820;
const float POLY_3             = 0.1666647195816;
const float POLY_4             = 0.4999998509884;
const float POLY_56            = 1.0000000000000;
const int32_t MAXIMUM_EXPONENT = 0x3F800000;

#pragma omp parallel for num_threads({th})
for (int i = 0; i < {batch}; i++) {{
    const float *input = {input_name} + i * {axis_len};
    float *output = {output_name} + i * {axis_len};

    float max = -INFINITY;
    for (int j = 0; j < {axis_len}; j++) {{
        max = fmaxf(input[j], max);
    }}

    float sum = 0.0;
    #pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        const float val0 = fmaxf(input[j] - max, LOWER_RANGE);
        const float biased = fmaf(val0, LOG2RECIPROCAL, ROUNDING_BIAS);
        const float m = biased - ROUNDING_BIAS;
        const float val1 = fmaf(m, LOG2HIGH, val0);
        const float val2 = fmaf(m, LOG2LOW, val1);
        const int32_t normal = (*(int *)&biased) << 23;
        const int32_t normal2 = normal + MAXIMUM_EXPONENT;
        const float p0 = POLY_0;
        const float p1 = fmaf(p0, val2, POLY_1);
        const float p2 = fmaf(p1, val2, POLY_2);
        const float p3 = fmaf(p2, val2, POLY_3);
        const float p4 = fmaf(p3, val2, POLY_4);
        const float p5 = fmaf(p4, val2, POLY_56);
        const float p6 = fmaf(p5, val2, POLY_56);
        const float p7 = p6 * (*(float *)&normal2);
        sum += p7;
        output[j] = p7;
    }}

    float recip_sum = 1.0 / sum;
    #pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        output[j] = output[j] * recip_sum;
    }}
}}",
            th = self.intra_op_num_threads,
            batch = output.dims.total_elems() / axis_len,
        );

        Ok(kernel)
    }

    fn translate_batch_norm(
        &mut self,
        bn: &BatchNormalization,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let data = inputs[0];
        let scale = inputs[1];
        let bias = inputs[2];
        let input_mean = inputs[3];
        let input_var = inputs[4];

        let data_name = &args[0];
        let scale_name = &args[1];
        let bias_name = &args[2];
        let input_mean_name = &args[3];
        let input_var_name = &args[4];
        let output_name = &args[inputs.len()..][0];

        assert!(!bn.training_mode, "Training mode is not supported.");
        assert!(data.elem_ty.is_f32(), "Input data type must be f32.");
        assert_eq!(data.dims.len(), 4, "Input data rank must be 4.");
        assert_eq!(scale.dims.len(), 1, "Scale rank must be 1.");
        assert_eq!(bias.dims.len(), 1, "Bias rank must be 1.");
        assert_eq!(input_mean.dims.len(), 1, "Input mean rank must be 1.");
        assert_eq!(input_var.dims.len(), 1, "Input var rank must be 1.");

        let num_batch = data.dims[0];
        let num_channel = data.dims[1];
        let num_dim0 = data.dims[2];
        let num_dim1 = data.dims[3];
        let data_strides = data.dims.strides();

        let kernel = format!(
            "#pragma omp parallel for num_threads({th})
for (int i = 0; i < {num_batch}; i++) {{
    const float *_data = {data_name} + i * {data_strides_0};
    float *output = {output_name} + i * {data_strides_0};
    for (int j = 0; j < {num_channel}; j++) {{
        const float _mean = {input_mean_name}[j];
        const float _var = {input_var_name}[j];
        const float _inv_var = 1.0 / sqrtf(_var + {epsilon});
        const float _scale = {scale_name}[j];
        const float _bias = {bias_name}[j];
        for (int k = 0; k < {num_dim0}; k++) {{
            for (int l = 0; l < {num_dim1}; l++) {{
                const float x = _data[j * {data_strides_1} + k * {data_strides_2} + l];
                output[j * {data_strides_1} + k * {data_strides_2} + l] =
                    (x - _mean) * _inv_var * _scale + _bias;
            }}
        }}
    }}
}}",
            th = self.intra_op_num_threads,
            epsilon = bn.epsilon,
            data_strides_0 = data_strides[0],
            data_strides_1 = data_strides[1],
            data_strides_2 = data_strides[2],
        );

        Ok(kernel)
    }

    fn translate_layer_norm(
        &mut self,
        ln: &LayerNormalization,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
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
            "#pragma omp parallel for num_threads({th})
for (int i = 0; i < {batch}; i++) {{
    float sum = 0.0;
    const float *data = {data_name} + i * {axis_len};
    float *output = {output_name} + i * {axis_len};
#pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        sum = sum + data[j];
    }}
    const float mean = sum * {inv_axis_len};
    float sum_squares = 0.0;
#pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        const float x = data[j] - mean;
        output[j] = x;
        sum_squares += x * x;
    }}
    const float inv_mean = 1.0 / sqrtf(sum_squares * {inv_axis_len} + {epsilon});
#pragma clang loop vectorize(enable)
    for (int j = 0; j < {axis_len}; j++) {{
        output[j] = output[j] * inv_mean * {scale_name}[j] + {bias_name}[j];
    }}
}}",
            th = self.intra_op_num_threads,
            batch = data.dims.total_elems() / axis_len,
            inv_axis_len = (axis_len as f32).recip(),
            epsilon = ln.epsilon,
        );

        Ok(kernel)
    }

    fn translate_gelu(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];
        let _input = inputs[0];
        let output = &outputs[0];

        let kernel = format!(
            "
const float B           = 0.7978845608028654;   // sqrt(2.0 / PI)
const float C           = 0.035677408136300125; // 0.044715 * sqrt(2.0 / PI)
const float LOWER_RANGE = -9;
const float UPPER_RANGE = 9;
const float ALPHA_13    = -2.76076847742355e-16;
const float ALPHA_11    = 2.00018790482477e-13;
const float ALPHA_9     = -8.60467152213735e-11;
const float ALPHA_7     = 5.12229709037114e-08;
const float ALPHA_5     = 1.48572235717979e-05;
const float ALPHA_3     = 6.37261928875436e-04;
const float ALPHA_1     = 4.89352455891786e-03;
const float BETA_6      = 1.19825839466702e-06;
const float BETA_4      = 1.18534705686654e-04;
const float BETA_2      = 2.26843463243900e-03;
const float BETA_0      = 4.89352518554385e-03;

#pragma omp parallel for num_threads({th})
#pragma clang loop vectorize(enable)
for (int i = 0; i < {size}; i++) {{
#define clamp(x, min, max) fminf(fmaxf(x, min), max)
// #define clamp(x, min, max) (x)
    const float x = {input_name}[i];
    const float y =
        clamp(x * fmaf(C * x, x, B), LOWER_RANGE, UPPER_RANGE);
    const float y_squared = y * y;

    float p = fmaf(y_squared, ALPHA_13, ALPHA_11);
    p = fmaf(p, y_squared, ALPHA_9);
    p = fmaf(p, y_squared, ALPHA_7);
    p = fmaf(p, y_squared, ALPHA_5);
    p = fmaf(p, y_squared, ALPHA_3);
    p = fmaf(p, y_squared, ALPHA_1);
    p = p * y;

    float q = fmaf(y_squared, BETA_6, BETA_4);
    q = fmaf(q, y_squared, BETA_2);
    q = fmaf(q, y_squared, BETA_0);

    float z = p / q;
    z = (z + 1.f) * (x * 0.5f);

    {output_name}[i] = z;
#undef clamp
}}",
            th = self.intra_op_num_threads,
            size = output.dims.total_elems(),
        );

        Ok(kernel)
    }

    fn translate_split(
        &mut self,
        split: &Split,
        args: &[String],
        inputs: &[&TypedFixedShape],
        _outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let opset_version = self.model.opset_version;

        let axis = if split.axis < 0 {
            inputs[0].dims.len() as i64 + split.axis
        } else {
            split.axis
        } as usize;
        assert_eq!(axis, inputs[0].dims.len() - 1);
        assert!(inputs[0].elem_ty.is_f32());
        let axis_len = *inputs[0].dims.as_slice().last().unwrap();

        let kernel = if opset_version >= 13 {
            let input_name = &args[0];
            let split_name = &args[1];
            let output_names = &args[inputs.len()..];
            let size = inputs[0].dims.total_elems();
            let split_len = inputs[1].dims.total_elems();

            format!(
                "int offsets[{split_len}] = {{0}};
float *outputs[] = {{ {outputs} }};
for (int i = 0; i < {size} / {axis_len}; i++) {{
    int s = 0;
    for (int j = 0; j < {split_len}; j++) {{
        const int sp = {split_name}[j];
        float *output = outputs[j];
        memcpy(output + offsets[j], {input_name} + i * {axis_len} + s, sp * sizeof(float));
        offsets[j] += sp;
        s += sp;
    }}
}}",
                outputs = output_names.join(", "),
            )
        } else {
            let input_name = &args[0];
            let output_names = &args[inputs.len()..];
            let size = inputs[0].dims.total_elems();
            let split_len = split.split.len();

            format!(
                "int offsets[{split_len}] = {{0}};
int splits[{split_len}] = {{ {splits} }};
float *outputs[] = {{ {outputs} }};
for (int i = 0; i < {size} / {axis_len}; i++) {{
    int s = 0;
    for (int j = 0; j < {split_len}; j++) {{
        const int sp = splits[j];
        float *output = outputs[j];
        memcpy(output + offsets[j], {input_name} + i * {axis_len} + s, sp * sizeof(float));
        offsets[j] += sp;
        s += sp;
    }}
}}",
                outputs = output_names.join(", "),
                splits = split
                    .split
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        };

        Ok(kernel)
    }

    fn translate_slice(
        &mut self,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        assert_eq!(inputs.len(), 4); // data, starts, ends, axes
                                     // TODO: Support steps

        let shape_data = inputs[0];
        let shape_starts = inputs[1];
        let shape_ends = inputs[2];
        let shape_axes = inputs[3];
        let shape_output = &outputs[0];

        let data = &args[0];
        let starts = &args[1];
        let ends = &args[2];
        let axes = &args[3];
        let output = &args[4];

        // Very limited support for now
        // TODO: Add support for wider variety of shapes
        assert_eq!(shape_data.dims.len(), 3);
        assert_eq!(shape_output.dims.len(), 3);
        assert_eq!(shape_starts.dims.len(), 1);
        assert_eq!(shape_ends.dims.len(), 1);
        assert_eq!(shape_axes.dims.len(), 1);
        assert_eq!(shape_starts.dims[0], 1);
        assert_eq!(shape_ends.dims[0], 1);
        assert_eq!(shape_axes.dims[0], 1);

        let dims = shape_data
            .dims
            .as_slice()
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let num_dims = 3;
        let num_axes = 1;
        let kernel = format!(
            r#"

int64_t dims  [{num_dims}] = {{ {dims} }};
int64_t steps [{num_dims}] = {{ 1, 1, 1 }};
int64_t starts[{num_axes}] = {{ {starts}[0] }};
int64_t ends  [{num_axes}] = {{ {ends}[0] }};
int64_t axes  [{num_axes}] = {{ {axes}[0] }};

int64_t effective_starts[{num_dims}];
int64_t effective_ends[{num_dims}];
int64_t effective_steps[{num_dims}];

for (int i = 0; i < {num_dims}; i++) {{
    effective_starts[i] = 0;
    effective_ends[i] = dims[i];
    effective_steps[i] = 1;
}}

for (int i = 0; i < {num_axes}; i++) {{
    if (axes[i] < 0) axes[i] += {num_dims};
    if (starts[i] < 0) starts[i] += dims[axes[i]];
    if (ends[i] < 0) ends[i] += dims[axes[i]];
    effective_starts[axes[i]] = starts[i];
    effective_ends[axes[i]] = ends[i];
    effective_steps[axes[i]] = steps[i];
}}

assert(effective_steps[0] == 1);
assert(effective_steps[1] == 1);
assert(effective_steps[2] == 1);

int64_t idx = 0;
for (int i = effective_starts[0]; i < effective_ends[0]; i += /*effective_steps[0]=*/ 1) {{
    for (int j = effective_starts[1]; j < effective_ends[1]; j += /*effective_steps[1]=*/ 1) {{
        #pragma clang loop vectorize(enable)
        for (int k = effective_starts[2]; k < effective_ends[2]; k += /*effective_steps[2]=*/ 1)
            {output}[idx++] = {data}[i * dims[1] * dims[2] + j * dims[2] + k];
    }}
}}
"#
        );

        Ok(kernel)
    }

    fn translate_cast(
        &mut self,
        cast: &Cast,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_name = &args[0];
        let output_name = &args[1];
        let input = inputs[0];
        let output = &outputs[0];
        assert_eq!(input.dims.total_elems(), output.dims.total_elems());

        let to = get_c_type(cast.to);
        let size = output.dims.total_elems();

        let kernel = format!(
            "for (int i = 0; i < {size}; i++) {{
    {output_name}[i] = ({to}){input_name}[i];
}}"
        );

        Ok(kernel)
    }

    fn translate_resize(
        &mut self,
        resize: &Resize,
        args: &[String],
        inputs: &[&TypedFixedShape],
        outputs: &[TypedFixedShape],
    ) -> Result<String, SessionError> {
        let input_names = &args[0..inputs.len()];
        let output_names = &args[inputs.len()..];

        let input = inputs[0];
        // let sizes = inputs[1];
        let output = &outputs[0];

        assert!(matches!(inputs.len(), 3 | 4));

        let batch_size = input.dims[0];
        let input_c = input.dims[1];
        let input_h = input.dims[2];
        let input_w = input.dims[3];
        let output_h = output.dims[2];
        let output_w = output.dims[3];
        let input_hw = input_h * input_w;
        let output_hw = output_h * output_w;
        let outer = batch_size * input_c;

        let scale = output_h as f32 / input_h as f32;

        if resize.mode == "nearest" {
            let kernel = format!("for (int i = 0; i < {outer}; i++) {{
    for (int h = 0; h < {output_h}; h++) {{
        const int ih = ((int)((float)h / (float){scale})) * {input_w};
        for (int w = 0; w < {output_w}; w++) {{
            const int iw = (float)w / (float){scale};
            {output_name}[i * {output_hw} + h * {output_w} + w] = {input_name}[i * {input_hw} + ih + iw];
        }}
    }}
}}",
                input_name = input_names[0],
                output_name = output_names[0]
            );
            Ok(kernel)
        } else {
            Err(SessionError::Message(
                format!("Resize: mode '{}' not supported", resize.mode).into(),
            ))
        }
    }

    fn create_file(&self, name: &str) -> Result<File, SessionError> {
        let path = self.target_dir.join(name);
        let file = File::create(path)?;
        Ok(file)
    }

    pub fn value_name(&self, id: ValueId) -> String {
        value_name(self.model, id)
    }
}

impl<'a> TranslationProduct<'a> {
    pub fn value_name(&self, id: ValueId) -> String {
        value_name(self.model, id)
    }
}

impl Default for CraneliftCtx {
    fn default() -> Self {
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.enable("is_pic").unwrap();
        let isa_builder = cranelift_codegen::isa::lookup(Triple::host()).unwrap();
        let isa = isa_builder
            .finish(cranelift_codegen::settings::Flags::new(flag_builder))
            .unwrap();
        let builder =
            ObjectBuilder::new(isa, "builder", cranelift_module::default_libcall_names()).unwrap();
        let module = ObjectModule::new(builder);
        Self {
            ctx: module.make_context(),
            module,
            num_vars: 0,
        }
    }
}

impl CraneliftCtx {
    pub fn new_var(&mut self) -> Variable {
        let var = Variable::new(self.num_vars);
        self.num_vars += 1;
        var
    }

    pub fn create_var(&mut self, ty: Type, init: Value, builder: &mut FunctionBuilder) -> Variable {
        let var = self.new_var();
        builder.declare_var(var, ty);
        builder.def_var(var, init);
        var
    }
}

impl Regions {
    fn alloc(&mut self, id: ValueId, size: usize) -> Range<usize> {
        let region = self.find_first_free_region(size);
        self.val_to_start.insert(id, region.start);
        self.start_to_region.insert(region.start, region.clone());
        self.max_allocated = self
            .max_allocated
            .max(self.start_to_region.last_key_value().unwrap().1.end);
        region
    }

    fn free(&mut self, id: ValueId) {
        if let Some(start) = self.val_to_start.remove(&id) {
            self.start_to_region.remove(&start).unwrap();
        }
    }

    fn find_first_free_region(&self, size: usize) -> Range<usize> {
        // Sorted by start offset
        let regions = self.start_to_region.values().collect::<Vec<_>>();
        fn roundup(x: usize) -> usize {
            let mask = 32 - 1;
            (x + mask) & !mask
        }
        match self.start_to_region.len() {
            0 => 0..roundup(size),
            1 if regions[0].start >= size => 0..roundup(size),
            1 => roundup(regions[0].end)..roundup(regions[0].end + size),
            _ => {
                for r in regions.windows(2) {
                    let [cur, next] = [r[0], r[1]];
                    if next.start - cur.end >= size {
                        return roundup(cur.end)..roundup(cur.end + size);
                    }
                }
                let last = regions.last().unwrap();
                roundup(last.end)..roundup(last.end + size)
            }
        }
    }
}

const fn get_clif_type(t: TensorElemType) -> Type {
    match t {
        TensorElemType::F32 => F32,
        TensorElemType::I32 => I32,
        TensorElemType::I64 => I64,
        TensorElemType::Bool => I8,
    }
}

fn value_name(model: &Model, id: ValueId) -> String {
    let value = &model.graph.values.inner()[id];
    escape_name(value.name.clone().map_or_else(
        || format!("Value_noname_{}", id.index()),
        |name| format!("Value_{name}_{}", id.index()),
    ))
}

fn escape_name(s: impl Into<String>) -> String {
    fn add_prefix(s: String) -> String {
        if s.starts_with(|c: char| c.is_ascii_digit()) {
            format!("_{s}")
        } else {
            s
        }
    }
    add_prefix(
        s.into()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect(),
    )
}

fn compute_sha1_from_files(paths: &[PathBuf]) -> Option<[u8; 20]> {
    let mut hasher = Sha1::new();
    for path in paths {
        let mut file = File::open(path).ok()?;
        std::io::copy(&mut file, &mut hasher).ok()?;
    }
    let hash = hasher.finalize();
    hash.get(..20)?.try_into().ok()
}

const fn get_c_type(ty: TensorElemType) -> &'static str {
    match ty {
        TensorElemType::F32 => "float",
        TensorElemType::I32 => "int32_t",
        TensorElemType::I64 => "int64_t",
        TensorElemType::Bool => "unsigned char",
    }
}

fn get_project_root() -> Option<PathBuf> {
    let path = std::env::current_dir().ok()?;
    let path_ancestors = path.as_path().ancestors();

    for p in path_ancestors {
        let has_cargo = std::fs::read_dir(p)
            .ok()?
            .any(|p| p.unwrap().file_name() == *"Cargo.lock");
        if has_cargo {
            return Some(PathBuf::from(p));
        }
    }

    None
}

fn find_path_from_project_root(path: &str) -> Option<PathBuf> {
    glob::glob(get_project_root()?.join(path).to_str()?)
        .ok()?
        .next()?
        .ok()
}
