import time
import logging
import os

import onnx
import altius_py
import torch

from PIL import Image
from torchvision import transforms


def main():
    logging.basicConfig(level=logging.INFO)

    image = Image.open("../models/cat.png")
    labels = open("../models/imagenet_classes.txt").readlines()

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    input = preprocess(image)
    input = input.unsqueeze(0).numpy()

    onnx_path = "../models/deit.onnx"

    if not os.path.exists(onnx_path):
        import onnxsim
        from transformers import ViTImageProcessor, ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "facebook/deit-small-patch16-224"
        )
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_path)
        simplified_model, success = onnxsim.simplify(onnx_path)
        assert success
        onnx.save(simplified_model, onnx_path)

    altius_model = altius_py.InferenceSession(
        onnx_path, intra_op_num_threads=1, enable_profile=True
    )

    with torch.no_grad():
        for i in range(1):
            output = altius_model.run(None, {"input.1": input})
            pred = torch.tensor(output).reshape((-1,)).argsort().numpy()[::-1][:5]
            top5 = [labels[i].strip() for i in pred]
            print(top5)


if __name__ == "__main__":
    main()
