# Convert float32 values into float16 values in ONNX model

import argparse

import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    f16_model = convert_float_to_float16(model)
    onnx.save(f16_model, args.output)
