extern crate altius_core;
extern crate altius_session;

use altius_core::optimize;
use altius_core::tensor::TensorElemTypeExt;
use altius_core::value::ValueId;
use altius_core::{model::Model, tensor::Tensor};
use altius_session::interpreter::Interpreter;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use numpy::ndarray::ArrayD;
use numpy::{Element, PyArrayDyn, PyReadonlyArrayDyn};

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

#[pyfunction(enable_profiling = false, intra_op_num_threads = 1)]
fn session<'a>(
    model: &'a mut PyModel,
    enable_profiling: bool,
    intra_op_num_threads: usize,
) -> PyResult<PySession> {
    let mut model =
        unsafe { std::mem::transmute::<&'a mut Model, &'static mut Model>(&mut model.0) };
    optimize::gelu_fusion::fuse_gelu(&mut model);
    Ok(PySession(
        Interpreter::new(model)
            .with_profiling(enable_profiling)
            .with_intra_op_num_threads(intra_op_num_threads),
    ))
}

#[pymethods]
impl PySession {
    pub fn run<'a>(
        &mut self,
        py: Python<'a>,
        inputs: &PyDict,
    ) -> PyResult<Vec<&'a PyArrayDyn<f32>>> {
        fn create_input<T: Element + TensorElemTypeExt>(
            model: &Model,
            name: String,
            val: PyReadonlyArrayDyn<T>,
        ) -> PyResult<(ValueId, Tensor)> {
            let val_id = model
                .lookup_named_value(&name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("Input '{}' not found", name)))?;
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
            .map_err(|_| PyRuntimeError::new_err("Inference failed".to_string()))?
        {
            let arr =
                ArrayD::from_shape_vec(out.dims().as_slice().to_vec(), out.data::<f32>().to_vec())
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to create output array: {:?}", e))
                    })?;
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
