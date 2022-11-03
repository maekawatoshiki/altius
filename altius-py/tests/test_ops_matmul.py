import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_matmul_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_matmul(os.path.join(tmpdir, "model.onnx"), [5, 10], [10, 15], [5, 15])


def test_matmul_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_matmul(os.path.join(tmpdir, "model.onnx"), [3, 5, 10], [10, 15], [3, 5, 15])


def test_matmul_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_matmul(
            os.path.join(tmpdir, "model.onnx"), [3, 5, 10], [3, 10, 15], [3, 5, 15]
        )


def test_matmul_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_matmul(os.path.join(tmpdir, "model.onnx"), [1, 5, 10], [10, 15], [1, 5, 15])


def op_matmul(
    filepath,
    shape_x,
    shape_y,
    shape_z,
):
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
    ]
    outputs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape_z)]
    nodes = [
        helper.make_node(
            "MatMul",
            ["x", "y"],
            ["z"],
        )
    ]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape_x).astype(np.float32)
    y = np.random.random_sample(shape_y).astype(np.float32)
    inputs = {"x": x, "y": y}
    expected = ort_sess.run(None, inputs)
    actual = altius_sess.run(None, inputs)

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
