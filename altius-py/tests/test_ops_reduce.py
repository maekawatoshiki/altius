import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_reduce_mean_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_reduce(
            os.path.join(tmpdir, "model.onnx"),
            "ReduceMean",
            [1, 50, 70],
            [1, 50, 1],
            axes=[-1],
        )


def test_reduce_max_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_reduce(
            os.path.join(tmpdir, "model.onnx"),
            "ReduceMax",
            [1, 50, 70],
            [],
            keepdims=0,
        )


def op_reduce(filepath, op_type, shape_x, shape_y, **kwargs):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [helper.make_node(op_type, ["x"], ["y"], **kwargs)]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

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
