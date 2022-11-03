import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


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
