import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_transpose_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_concat(
            os.path.join(tmpdir, "model.onnx"),
            [50, 12, 64],
            [12, 64, 50],
            perm=[1, 2, 0],
        )


def test_transpose_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_concat(
            os.path.join(tmpdir, "model.onnx"),
            [12, 64],
            [64, 12],
            perm=[1, 0],
        )


def test_transpose_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_concat(
            os.path.join(tmpdir, "model.onnx"),
            [12, 64, 3, 5],
            [3, 64, 12, 5],
            perm=[2, 1, 0, 3],
        )


def op_concat(filepath, shape_x, shape_y, **kwargs):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [helper.make_node("Transpose", ["x"], ["y"], **kwargs)]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.checker.check_model(model)
    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape_x).astype(np.float32)
    y = np.random.random_sample(shape_y).astype(np.float32)
    inputs = {"x": x}
    expected = ort_sess.run(None, inputs)
    actual = altius_sess.run(None, inputs)

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
