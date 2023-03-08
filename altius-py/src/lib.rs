extern crate altius_core;
extern crate altius_session;

use altius_core::optimize;
use altius_core::tensor::{TensorElemType, TensorElemTypeExt};
use altius_core::value::ValueId;
use altius_core::{model::Model, tensor::Tensor};
use altius_session::interpreter::{InterpreterSession, InterpreterSessionBuilder};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use numpy::ndarray::ArrayD;
use numpy::{Element, IntoPyArray, PyReadonlyArrayDyn};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyModel(pub Model);

#[pyclass]
#[repr(transparent)]
pub struct PySession(pub InterpreterSession);

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

#[pyfunction(enable_profiling = false, intra_op_num_threads = 1)]
fn session(
    model: PyModel,
    enable_profiling: bool,
    intra_op_num_threads: usize,
) -> PyResult<PySession> {
    let mut model = model.0;
    optimize::layer_norm_fusion::fuse_layer_norm(&mut model);
    optimize::gelu_fusion::fuse_gelu(&mut model);
    Ok(PySession(
        InterpreterSessionBuilder::new(model)
            .with_profiling_enabled(enable_profiling)
            .with_intra_op_num_threads(intra_op_num_threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    ))
}

#[pymethods]
impl PySession {
    pub fn run(&mut self, py: Python, inputs: &PyDict) -> PyResult<Vec<Py<PyAny>>> {
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

        let mut new_inputs = vec![];
        for (i, item) in inputs.items().iter().enumerate() {
            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<f32>)>() {
                new_inputs.push(create_input(self.0.model(), name, val)?);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<i64>)>() {
                new_inputs.push(create_input(self.0.model(), name, val)?);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<i32>)>() {
                new_inputs.push(create_input(self.0.model(), name, val)?);
                continue;
            }

            if let Ok((name, val)) = item.extract::<(String, PyReadonlyArrayDyn<bool>)>() {
                new_inputs.push(create_input(self.0.model(), name, val)?);
                continue;
            }

            return Err(PyRuntimeError::new_err(format!(
                "Input {i} unsupported type"
            )));
        }

        let mut outputs = vec![];
        for out in self
            .0
            .run(new_inputs)
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

/// A Python module implemented in Rust.
#[pymodule]
fn altius_py(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(session, m)?)?;
    Ok(())
}
