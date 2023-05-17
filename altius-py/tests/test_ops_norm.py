import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_layer_norm_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_layer_norm(os.path.join(tmpdir, "model.onnx"), [1, 20, 10])


def op_layer_norm(filepath, shape, **kwargs):
    shape_scale = [1] * (len(shape) - 1) + [shape[-1]]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, shape_scale),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT, shape_scale),
    ]
    outputs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)]
    nodes = [
        helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["z"], **kwargs)
    ]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

    for backend in ["interpreter", "cpu"]:
        altius_sess = altius_py.InferenceSession(filepath, backend=backend)

        x = np.random.random_sample(shape).astype(np.float32)
        scale = np.random.random_sample(shape_scale).astype(np.float32)
        bias = np.random.random_sample(shape_scale).astype(np.float32)
        inputs = {"x": x, "scale": scale, "bias": bias}
        expected = ort_sess.run(None, inputs)
        actual = altius_sess.run(None, inputs)

        for expected, actual in zip(expected, actual):
            assert np.allclose(expected, actual, rtol=1e-4, atol=1e-5)
