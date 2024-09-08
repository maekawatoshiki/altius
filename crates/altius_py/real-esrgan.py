import altius_py
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import os, random
from matplotlib import pyplot as plt
import onnxruntime as ort
import logging
from torchvision.transforms.functional import to_pil_image
import torch


def main():
    logging.basicConfig(level=logging.INFO)
    image = Image.open("../../models/cat.png").convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input = preprocess(image)
    input = input.unsqueeze(0).numpy()
    print(input.shape)

    path = "../../models/realesrgan_256x256.onnx"
    sess = altius_py.InferenceSession(
        path, intra_op_num_threads=32, enable_profile=True
    )
    # sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    # sess_options = ort.SessionOptions()
    # # sess_options.enable_profiling = True
    # sess_options.intra_op_num_threads = 16
    # sess_options.inter_op_num_threads = 1
    # sess = ort.InferenceSession(
    #     path,
    #     providers=["CPUExecutionProvider"],
    #     # providers=["CUDAExecutionProvider"],
    #     sess_options=sess_options,
    # )

    inputs = {"input.1": input}
    start = time.time()
    output = sess.run(None, inputs)[0]
    print(f"elapsed: {time.time() - start}")

    # print(output.shape)
    # print(output.max())
    # print(output.min())
    img = to_pil_image(torch.tensor(output.clip(0, 1)).squeeze())

    img.save("a.png")
    img.show()


if __name__ == "__main__":
    main()
