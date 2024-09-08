import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_gather_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_gather(
            os.path.join(tmpdir, "model.onnx"),
            [1, 5, 10],
            2,
            [1, 1, 10],
            axis=1,
        )


def test_gather_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_gather(
            os.path.join(tmpdir, "model.onnx"),
            [5, 10],
            [1, 3],
            [2, 10],
            axis=0,
        )


def op_gather(filepath, shape_x, indices, shape_z, **kwargs):
    shape_y = [] if isinstance(indices, int) else [1, len(indices)]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("y", TensorProto.INT64, shape_y),
    ]
    outputs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape_z)]
    nodes = [helper.make_node("Gather", ["x", "y"], ["z"], **kwargs)]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

    for backend in ["interpreter", "cpu"]:
        altius_sess = altius_py.InferenceSession(filepath, backend="cpu")

        x = np.random.random_sample(shape_x).astype(np.float32)
        y = np.array(indices).astype(np.int64).reshape(shape_y)
        inputs = {"x": x, "y": y}
        expected = ort_sess.run(None, inputs)
        actual = altius_sess.run(None, inputs)

        for expected, actual in zip(expected, actual):
            assert np.allclose(expected, actual)
