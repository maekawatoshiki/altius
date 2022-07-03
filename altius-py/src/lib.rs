extern crate altius_core;
extern crate altius_interpreter;

use altius_core::model::Model;
use altius_interpreter::Interpreter2;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyModel(pub Model);

#[pyclass]
#[repr(transparent)]
pub struct PySession(pub Interpreter2<'static>);

#[pyfunction]
fn load(path: String) -> PyResult<PyModel> {
    altius_core::onnx::load_onnx(path).map_or(
        Err(PyRuntimeError::new_err("Failed to load a ONNX model")),
        |m| Ok(PyModel(m)),
    )
}

#[pyfunction]
fn session<'a>(model: &'a PyModel) -> PyResult<PySession> {
    let model = unsafe { std::mem::transmute::<&'a Model, &'static Model>(&model.0) };
    Ok(PySession(Interpreter2::new(model)))
}

/// A Python module implemented in Rust.
#[pymodule]
fn altius_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(session, m)?)?;
    Ok(())
}
