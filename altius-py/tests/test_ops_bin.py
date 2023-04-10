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
        op_bin(
            os.path.join(tmpdir, "model.onnx"),
            "Add",
            # Can not broadcast
            [1, 3, 28, 28],
            [3, 1, 2],
        )


def test_add_6():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [1, 12, 9, 9], [1, 1, 1, 9])


def test_add_7():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Add", [4, 1, 2], [3, 1], [4, 3, 2])


def test_sub_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 2])


def test_sub_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 2, 3, 4, 5, 6, 7])


def test_sub_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [128, 3, 224, 224])


@pytest.mark.xfail
def test_sub_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [1, 3, 28, 28], [3, 1, 1])


def test_sub_5():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Sub", [3, 28, 28], [3, 28, 1])


def test_mul_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [1, 2])


def test_mul_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [1, 2, 3, 4, 5, 6, 7])


def test_mul_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [128, 3, 224, 224])


def test_mul_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Mul", [3, 15, 15], [15])


def test_div_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [1, 2])


def test_div_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [1, 2, 3, 4, 5, 6, 7])


def test_div_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [128, 3, 224, 224])


def test_div_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [3, 28, 28], [3, 28, 1])


def test_div_5():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Div", [3, 28, 28], [1])


def test_pow_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Pow", [1, 2])


def test_pow_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Pow", [1, 2, 3, 4, 5, 6, 7])


def test_pow_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Pow", [128, 3, 8, 8])


def test_pow_4():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(os.path.join(tmpdir, "model.onnx"), "Pow", [128, 3, 8, 8], [1])


@pytest.mark.xfail
def test_greater_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(
            os.path.join(tmpdir, "model.onnx"),
            "Greater",
            [3, 4, 5],
            zdtype=TensorProto.BOOL,
        )


def test_greater_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        op_bin(
            os.path.join(tmpdir, "model.onnx"),
            "Greater",
            [3, 4, 5],
            shape_y=[1],
            zdtype=TensorProto.BOOL,
        )


def op_bin(
    filepath, op_type, shape_x, shape_y=None, shape_z=None, zdtype=TensorProto.FLOAT
):
    shape_y = shape_y if shape_y else shape_x
    shape_z = shape_z if shape_z else shape_x

    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
    ]
    outputs = [helper.make_tensor_value_info("z", zdtype, shape_z)]
    nodes = [helper.make_node(op_type, ["x", "y"], ["z"])]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    altius_sess = altius_py.InferenceSession(
        filepath, backend=os.getenv("ALTIUS_BACKEND", "interpreter")
    )

    x = np.random.random_sample(shape_x).astype(np.float32)
    y = np.random.random_sample(shape_y).astype(np.float32)
    expected = ort_sess.run(None, {"x": x, "y": y})
    actual = altius_sess.run(None, {"x": x, "y": y})

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
