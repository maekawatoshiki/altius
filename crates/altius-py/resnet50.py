import time
import os

os.environ["GOMP_CPU_AFFINITY"] = "0-7"
os.environ["OMP_WAIT_POLICY"] = "active"

from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torchvision.models import resnet50

import onnxruntime as ort
import altius_py


def main():
    model_path = "../../models/resnet50.onnx"

    if not os.path.exists(model_path):
        with torch.no_grad():
            model = resnet50(pretrained=True)
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224, dtype=torch.float32),
                model_path,
                verbose=True,
            )

    labels = open("../../models/imagenet_classes.txt").readlines()
    image = Image.open("../../models/cat.png")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input = preprocess(image).unsqueeze(0).numpy()

    use_ort = False
    if use_ort:
        sess = ort.InferenceSession(model_path)
    else:
        sess = altius_py.InferenceSession(
            model_path,
            intra_op_num_threads=4,
            backend="cpu",
        )

    inputs = {"input.1": input}
    for _ in range(10):
        start = time.time()
        output = sess.run(None, inputs)[0][0]
        print(f"Elapsed: {(time.time() - start) * 1000.0:.3f} [ms]")
    output = np.argsort(output)[::-1][:5]
    output = [labels[i].strip() for i in output]
    print(f"Top-5: {output}")


if __name__ == "__main__":
    main()
