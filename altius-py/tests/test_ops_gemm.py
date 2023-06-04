import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_gemm_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_gemm(os.path.join(tmpdir, "model.onnx"), [5, 10], [10, 15], [5, 15], [15])


def test_gemm_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_gemm(os.path.join(tmpdir, "model.onnx"), [5, 10], [10, 15], [5, 15], [5, 15])


def test_gemm_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_gemm(os.path.join(tmpdir, "model.onnx"), [5, 10], [10, 15], [5, 15])


# TODO
# def test_gemm_3():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         op_gemm(
#             os.path.join(tmpdir, "model.onnx"), [3, 5, 10], [3, 10, 15], [3, 5, 15]
#         )
#
#
# def test_gemm_4():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         op_gemm(os.path.join(tmpdir, "model.onnx"), [1, 5, 10], [10, 15], [1, 5, 15])


def op_gemm(filepath, shape_a, shape_b, shape_y, shape_c=None):
    inputs = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, shape_a),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, shape_b),
    ]
    if shape_c:
        inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, shape_c))
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [
        helper.make_node(
            "Gemm",
            ["a", "b", "c"] if shape_c else ["a", "b"],
            ["y"],
        )
    ]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

    for backend in ["interpreter", "cpu"]:
        altius_sess = altius_py.InferenceSession(filepath, backend=backend)

        a = np.random.random_sample(shape_a).astype(np.float32)
        b = np.random.random_sample(shape_b).astype(np.float32)
        inputs = {}
        if shape_c:
            c = np.random.random_sample(shape_c).astype(np.float32)
            inputs = {"a": a, "b": b, "c": c}
        else:
            inputs = {"a": a, "b": b}
        expected = ort_sess.run(None, inputs)
        actual = altius_sess.run(None, inputs)

        for expected, actual in zip(expected, actual):
            assert np.allclose(expected, actual)
