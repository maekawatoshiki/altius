import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_where_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_where(
            os.path.join(tmpdir, "model.onnx"),
            [1, 1, 10, 10],
            [1, 128, 10, 10],
            [1],
        )


def test_where_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_where(
            os.path.join(tmpdir, "model.onnx"),
            [1, 1, 1, 1],
            [1, 128, 1, 1],
            [1],
        )


def op_where(filepath, shape_c, shape_x, shape_y, **kwargs):
    inputs = [
        helper.make_tensor_value_info("c", TensorProto.BOOL, shape_c),
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
    ]
    outputs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape_x)]
    nodes = [helper.make_node("Where", ["c", "x", "y"], ["z"], **kwargs)]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.checker.check_model(model)
    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(filepath)

    c = np.random.choice(a=[False, True], size=shape_c)
    x = np.random.random_sample(shape_x).astype(np.float32)
    y = np.random.random_sample(shape_y).astype(np.float32)
    inputs = {"c": c, "x": x, "y": y}
    expected = ort_sess.run(None, inputs)
    actual = altius_sess.run(None, inputs)

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
