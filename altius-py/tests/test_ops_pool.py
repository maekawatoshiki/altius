import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_maxpool_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_maxpool(
            os.path.join(tmpdir, "model.onnx"),
            [1, 3, 224, 224],
            [1, 3, 112, 112],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
            auto_pad="NOTSET",
        )


def test_maxpool_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_maxpool(
            os.path.join(tmpdir, "model.onnx"),
            [1, 256, 20, 20],
            [1, 256, 20, 20],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            strides=[1, 1],
            ceil_mode=0,
        )


def op_maxpool(filepath, shape_x, shape_y, **kwargs):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [
        helper.make_node(
            "MaxPool",
            ["x"],
            ["y"],
            **kwargs,
        )
    ]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

    for backend in ["interpreter", "cpu"]:
        altius_sess = altius_py.InferenceSession(filepath, backend=backend)

        x = np.random.random_sample(shape_x).astype(np.float32)
        inputs = {"x": x}
        expected = ort_sess.run(None, inputs)
        actual = altius_sess.run(None, inputs)

        for expected, actual in zip(expected, actual):
            assert np.allclose(expected, actual)
