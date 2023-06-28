extern crate altius_core;
extern crate altius_session;

use std::collections::HashMap;

use altius_core::optimize;
use altius_core::tensor::{TensorElemType, TensorElemTypeExt};
use altius_core::value::ValueId;
use altius_core::{model::Model, tensor::Tensor};
use altius_session::cpu::{CPUSession, CPUSessionBuilder};
use altius_session::interpreter::{InterpreterSession, InterpreterSessionBuilder};
use altius_session::SessionError;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use numpy::ndarray::ArrayD;
use numpy::{Element, IntoPyArray, PyReadonlyArrayDyn};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyModel(pub Model);

#[pyclass]
#[repr(transparent)]
pub struct PyInterpreterSession(pub InterpreterSession);

#[pyclass]
#[repr(transparent)]
pub struct PyCPUSession(pub CPUSession);

#[pyfunction]
fn load(path: String) -> PyResult<PyModel> {
    altius_core::onnx::load_onnx(path).map_or_else(
        |e| {
            Err(PyRuntimeError::new_err(format!(
                "Failed to load a ONNX model: {e:?}"
            )))
        },
        |m| Ok(PyModel(m)),
    )
}

#[pyfunction(
    enable_profiling = false,
    intra_op_num_threads = 1,
    backend = "\"interpreter\".to_string()"
)]
fn session(
    py: Python,
    model: PyModel,
    enable_profiling: bool,
    intra_op_num_threads: usize,
    backend: String,
) -> PyResult<Py<PyAny>> {
    let mut model = model.0;
    optimize::layer_norm_fusion::fuse_layer_norm(&mut model);
    optimize::gelu_fusion::fuse_gelu(&mut model);
    optimize::fast_gelu_fusion::fuse_fast_gelu(&mut model);
    if backend == "cpu" {
        optimize::elemwise_fusion::fuse_elemwise_ops(&mut model).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to fuse element-wise operations: {e:?}"))
        })?
    }

    match backend.as_str() {
        "interpreter" => Ok(PyInterpreterSession(
            py.allow_threads(|| {
                InterpreterSessionBuilder::new(model)
                    .with_profiling_enabled(enable_profiling)
                    .with_intra_op_num_threads(intra_op_num_threads)
                    .build()
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_py(py)),
        "cpu" => Ok(PyCPUSession(
            py.allow_threads(|| {
                CPUSessionBuilder::new(model)
                    .with_profiling_enabled(enable_profiling)
                    .with_intra_op_num_threads(intra_op_num_threads)
                    .build()
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_py(py)),
        _ => PyResult::Err(PyRuntimeError::new_err(format!(
            "Unknown backend: {backend}"
        ))),
    }
}

trait Session {
    fn model(&self) -> &Model;
    fn _run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError>;

    fn run(&self, py: Python, inputs: &PyDict) -> PyResult<Vec<Py<PyAny>>> {
        fn create_input<T: Element + TensorElemTypeExt>(
            model: &Model,
            name: String,
            val: PyReadonlyArrayDyn<T>,
        ) -> PyResult<(ValueId, Tensor)> {
            let val_id = model
                .lookup_named_value(&name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("Input '{name}' not found")))?;
            Ok((
                val_id,
                Tensor::new(
                    val.shape().to_vec().into(),
                    val.as_slice()
                        .map_err(|_| PyRuntimeError::new_err("Array not contiguous"))?
                        .to_vec(),
                ),
            ))
        }

        let mut new_inputs = HashMap::<ValueId, Tensor>::default();
        for (i, item) in inputs.items().iter().enumerate() {
            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<f32>)>() {
                let (key, val) = create_input(self.model(), name, val)?;
                new_inputs.insert(key, val);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<i64>)>() {
                let (key, val) = create_input(self.model(), name, val)?;
                new_inputs.insert(key, val);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<i32>)>() {
                let (key, val) = create_input(self.model(), name, val)?;
                new_inputs.insert(key, val);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<bool>)>() {
                let (key, val) = create_input(self.model(), name, val)?;
                new_inputs.insert(key, val);
                continue;
            }

            return Err(PyRuntimeError::new_err(format!(
                "Input {i} unsupported type"
            )));
        }

        let new_inputs = self
            .model()
            .inputs
            .iter()
            .map(|i| new_inputs.remove(i).unwrap())
            .collect::<Vec<_>>();
        let mut outputs = vec![];
        for out in self
            ._run(new_inputs)
            .map_err(|e| PyRuntimeError::new_err(format!("Inference failed: {e}")))?
        {
            macro_rules! arr {
                ($t:tt) => {
                    ArrayD::from_shape_vec(
                        out.dims().as_slice().to_vec(),
                        out.data::<$t>().to_vec(),
                    )
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to create output array: {e:?}"))
                    })?
                    .into_pyarray(py)
                    .to_object(py)
                };
            }
            let arr = match out.elem_ty() {
                TensorElemType::F32 => arr!(f32),
                TensorElemType::I32 => arr!(i32),
                TensorElemType::I64 => arr!(i64),
                TensorElemType::Bool => arr!(bool),
            };
            outputs.push(arr);
        }
        Ok(outputs)
    }
}

impl Session for PyInterpreterSession {
    fn model(&self) -> &Model {
        self.0.model()
    }

    fn _run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError> {
        self.0.run(inputs)
    }
}

impl Session for PyCPUSession {
    fn model(&self) -> &Model {
        self.0.model()
    }

    fn _run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, SessionError> {
        self.0.run(inputs)
    }
}

#[pymethods]
impl PyInterpreterSession {
    fn run(&self, py: Python, inputs: &PyDict) -> PyResult<Vec<Py<PyAny>>> {
        Session::run(self, py, inputs)
    }
}

#[pymethods]
impl PyCPUSession {
    fn run(&self, py: Python, inputs: &PyDict) -> PyResult<Vec<Py<PyAny>>> {
        Session::run(self, py, inputs)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn altius_py(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(session, m)?)?;
    Ok(())
}
