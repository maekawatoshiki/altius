import altius_py
import onnxruntime as ort
import onnx
import tempfile
import pytest
import os
import numpy as np
from onnx import helper, ValueInfoProto, TensorProto


@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_1(bias):
    with tempfile.TemporaryDirectory() as tmpdir:
        op_conv2d(
            os.path.join(tmpdir, "model.onnx"),
            [1, 3, 224, 224],
            [16, 3, 3, 3],
            [1, 16, 112, 112],
            bias=bias,
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )


@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_2(bias):
    with tempfile.TemporaryDirectory() as tmpdir:
        op_conv2d(
            os.path.join(tmpdir, "model.onnx"),
            [1, 16, 112, 112],
            [16, 1, 3, 3],
            [1, 16, 112, 112],
            bias=bias,
            group=16,
            pads=[1, 1, 1, 1],
        )


def op_conv2d(filepath, shape_x, shape_w, shape_y, bias=False, **kwargs):
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
        helper.make_tensor_value_info("w", TensorProto.FLOAT, shape_w),
    ]
    if bias:
        inputs.append(
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [shape_w[0]])
        )

    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)]
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w", "b"] if bias else ["x", "w"],
            ["y"],
            kernel_shape=[shape_w[2], shape_w[3]],
            **kwargs,
        )
    ]
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(graph)

    onnx.save(model, filepath)
    ort_sess = ort.InferenceSession(filepath)
    altius_sess = altius_py.InferenceSession(filepath)

    x = np.random.random_sample(shape_x).astype(np.float32)
    w = np.random.random_sample(shape_w).astype(np.float32)
    b = np.random.random_sample(shape_w[0]).astype(np.float32) if bias else None
    inputs = {"x": x, "w": w, "b": b} if bias else {"x": x, "w": w}
    expected = ort_sess.run(None, inputs)
    actual = altius_sess.run(None, inputs)

    for expected, actual in zip(expected, actual):
        assert np.allclose(expected, actual)
