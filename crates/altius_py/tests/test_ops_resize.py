import tempfile
import pytest
import os

import numpy as np

import onnxruntime as ort
import onnx
from onnx import helper, ValueInfoProto, TensorProto, numpy_helper
import altius_py


def test_resize_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_resize(
            os.path.join(tmpdir, "model.onnx"),
            [1, 256, 20, 20],
            [1, 256, 40, 40],
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            coordinate_transformation_mode="asymmetric",
            cubic_coeff_a=-0.75,
            mode="nearest",
            nearest_mode="floor",
        )


def op_resize(filepath, shape_x, shape_y, scales, **kwargs):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [
        helper.make_node(
            "Resize",
            ["x", "roi", "scales"],
            ["y"],
            **kwargs,
        )
    ]

    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(scales, name="scales")
    graph = helper.make_graph(
        nodes, "graph", inputs, outputs, initializer=[roi, scales]
    )
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape_x).astype(np.float32)
    inputs = {"x": x}
    expected = ort_sess.run(None, inputs)
    actual = altius_sess.run(None, inputs)

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
