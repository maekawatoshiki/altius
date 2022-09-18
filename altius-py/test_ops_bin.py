import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


def test_add_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [1, 2])


def test_add_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [1, 2, 3, 4, 5, 6, 7])


def test_add_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [128, 3, 224, 224])


def test_add_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [1, 3, 28, 28], [3, 1, 1])


@pytest.mark.xfail
def test_add_5():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_add(os.path.join(tmpdir, "model.onnx"), "Add", [1, 3, 28, 28], [3, 1, 2])


@pytest.mark.xfail
def test_sub_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 2])


@pytest.mark.xfail
def test_sub_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 2, 3, 4, 5, 6, 7])


@pytest.mark.xfail
def test_sub_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [128, 3, 224, 224])


@pytest.mark.xfail
def test_sub_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 3, 28, 28], [3, 1, 1])


def test_mul_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [1, 2])


def test_mul_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [1, 2, 3, 4, 5, 6, 7])


def test_mul_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [128, 3, 224, 224])


def test_div_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [1, 2])


def test_div_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [1, 2, 3, 4, 5, 6, 7])


def test_div_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [128, 3, 224, 224])


def op_bin(filepath, op_type, shape_x, shape_y=None, shape_z=None):
    shape_y = shape_y if shape_y else shape_x
    shape_z = shape_z if shape_z else shape_x

    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
    ]
    outputs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape_z)]
    nodes = [helper.make_node(op_type, ["x", "y"], ["z"])]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath)
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape_x).astype(np.float32)
    y = np.random.random_sample(shape_y).astype(np.float32)
    expected = ort_sess.run(None, {"x": x, "y": y})
    actual = altius_sess.run(None, {"x": x, "y": y})

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
