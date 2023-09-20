import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os

import numpy as np

import onnxsim
from onnx import helper, ValueInfoProto, TensorProto
from onnxscript import FLOAT, script, opset12 as op


def test_relu_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Relu", [1, 2])


def test_relu_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Relu", [3, 1, 28, 28])


def test_hardsigmoid_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "HardSigmoid", [1, 2])


def test_hardsigmoid_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "HardSigmoid", [3, 1, 28, 28])


def test_leakyrelu_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "LeakyRelu", [1, 2])


def test_leakyrelu_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "LeakyRelu", [3, 1, 28, 28])


def test_sigmoid_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Sigmoid", [1, 2])


def test_sigmoid_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Sigmoid", [3, 1, 28, 28])


def test_sigmoid_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Sigmoid", [3, 1, 7, 7])


@pytest.mark.xfail
def test_clip_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Clip", [1, 2])


@pytest.mark.xfail
def test_clip_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Clip", [3, 1, 28, 28])


def test_sqrt_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Sqrt", [1, 2])


def test_sqrt_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(os.path.join(tmpdir, "model.onnx"), "Sqrt", [3, 1, 28, 28])


def test_softmax_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(
            os.path.join(tmpdir, "model.onnx"),
            "Softmax",
            [1, 2, 3],
        )


def test_softmax_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(
            os.path.join(tmpdir, "model.onnx"),
            "Softmax",
            [3, 28, 28],
        )


def test_softmax_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(
            os.path.join(tmpdir, "model.onnx"),
            "Softmax",
            [128, 512, 512],
        )


def test_erf_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(
            os.path.join(tmpdir, "model.onnx"), "Erf", [1, 2, 3], atol=1e-1, rtol=1e-5
        )


def test_erf_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_elemwise(
            os.path.join(tmpdir, "model.onnx"), "Erf", [3, 28, 28], atol=1e-1, rtol=1e-5
        )


@script()
def gelu(x: FLOAT[1, 2, 3]) -> FLOAT[1, 2, 3]:
    half = op.Constant(
        value=onnx.helper.make_tensor("value", TensorProto.FLOAT, [1], [0.5])
    )
    one = op.Constant(
        value=onnx.helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0])
    )
    sqrt2 = op.Constant(
        value=onnx.helper.make_tensor("value", TensorProto.FLOAT, [1], [np.sqrt(2.0)])
    )
    return op.Mul(op.Mul(x, op.Add(op.Erf(op.Div(x, sqrt2)), one)), half)


def test_gelu_1():
    model = gelu.to_model_proto()
    model, ok = onnxsim.simplify(model)
    assert ok

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.onnx")
        onnx.save(model, filepath)
        ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
        altius_sess = altius_py.InferenceSession(filepath)

        x = np.random.random_sample([1, 2, 3]).astype(np.float32)
        expected = ort_sess.run(None, {"x": x})
        actual = altius_sess.run(None, {"x": x})

        for expected, actual in zip(expected, actual):
            assert np.allclose(expected, actual, atol=1e-1, rtol=1e-5)


def op_elemwise(filepath, op_type, shape, **kwargs):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
    nodes = [helper.make_node(op_type, ["x"], ["y"])]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape).astype(np.float32)
    expected = ort_sess.run(None, {"x": x})
    actual = altius_sess.run(None, {"x": x})

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual, **kwargs)
