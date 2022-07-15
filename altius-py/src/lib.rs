extern crate altius_core;
extern crate altius_interpreter;

use std::collections::HashMap;

use altius_core::{model::Model, tensor::Tensor};
use altius_interpreter::Interpreter;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use numpy::ndarray::ArrayD;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyModel(pub Model);

#[pyclass]
#[repr(transparent)]
pub struct PySession(pub Interpreter<'static>);

#[pyfunction]
fn load(path: String) -> PyResult<PyModel> {
    altius_core::onnx::load_onnx(path).map_or(
        Err(PyRuntimeError::new_err("Failed to load a ONNX model")),
        |m| Ok(PyModel(m)),
    )
}

#[pyfunction(enable_profiling = false)]
fn session<'a>(model: &'a PyModel, enable_profiling: bool) -> PyResult<PySession> {
    let model = unsafe { std::mem::transmute::<&'a Model, &'static Model>(&model.0) };
    Ok(PySession(
        Interpreter::new(model).with_profiling(enable_profiling),
    ))
}

#[pymethods]
impl PySession {
    pub fn run<'a>(
        &mut self,
        py: Python<'a>,
        inputs: &PyDict,
    ) -> PyResult<Vec<&'a PyArrayDyn<f32>>> {
        let map: HashMap<String, PyReadonlyArrayDyn<f32>> = inputs
            .extract()
            .map_err(|_| PyRuntimeError::new_err("Failed to extract inputs"))?;
        let mut inputs = vec![];
        for (name, val) in map {
            let val_id =
                self.0
                    .model()
                    .lookup_named_value(&name)
                    .ok_or(PyRuntimeError::new_err(format!(
                        "Input '{}' not found",
                        name
                    )))?;
            inputs.push((
                val_id,
                Tensor::new(
                    val.shape().to_vec().into(),
                    val.as_slice()
                        .map_err(|_| PyRuntimeError::new_err("Array not contiguous"))?
                        .to_vec(),
                ),
            ));
        }
        let mut outputs = vec![];
        for out in self.0.run(inputs) {
            let arr =
                ArrayD::from_shape_vec(out.dims().as_slice().to_vec(), out.data::<f32>().to_vec())
                    .map_err(|_| PyRuntimeError::new_err("Failed to create output array"))?;
            outputs.push(PyArrayDyn::from_array(py, &arr))
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
