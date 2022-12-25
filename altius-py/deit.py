import time
import logging
import os

import onnxsim
import altius_py
import onnx
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

import numpy as np
from PIL import Image


def main():
    logging.basicConfig(level=logging.INFO)

    image = Image.open("../models/cat.png")
    labels = open("../models/imagenet_classes.txt").readlines()

    feature_extractor = ViTImageProcessor.from_pretrained(
        "facebook/deit-small-patch16-224"
    )
    inputs = feature_extractor(image, return_tensors="pt").to("mps")

    onnx_path = "../models/deit.onnx"

    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-small-patch16-224"
    ).to("mps")
    if not os.path.exists(onnx_path):
        model = ViTForImageClassification.from_pretrained(
            "facebook/deit-small-patch16-224"
        ).to("mps")
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_path)
        simplified_model, success = onnxsim.simplify(onnx_path)
        assert success
        onnx.save(simplified_model, onnx_path)

    altius_model = altius_py.InferenceSession(
        onnx_path, intra_op_num_threads=8, enable_profile=True
    )
    input = inputs["pixel_values"].cpu().detach().numpy().reshape((1, 3, 224, 224))

    with torch.no_grad():
        for i in range(10):
            start = time.time()
            # output = model(**inputs).logits
            output = altius_model.run(None, {"input.1": input})
            print(time.time() - start)
            # print(output)
            pred = torch.tensor(output).reshape((-1,)).argsort().numpy()[::-1][:5]
            # pred = output.cpu().reshape((-1,)).argsort().numpy()[::-1][:5]
            top5 = [labels[i].strip() for i in pred]
            print(top5)


if __name__ == "__main__":
    main()
