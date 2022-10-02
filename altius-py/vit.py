import os
import random
import sys
import logging
import time

import numpy as np
from torchvision import transforms
from PIL import Image

import altius_py
import onnxruntime as ort


def main():
    logging.basicConfig(level=logging.INFO)

    labels = open("../models/imagenet_classes.txt").readlines()
    image = Image.open("../models/cat.png")

    preprocess = transforms.Compose(
        [
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input = preprocess(image)
    input = input.unsqueeze(0).numpy()

    # opt = ort.SessionOptions()
    # # opt.intra_op_num_threads = 1
    # # opt.inter_op_num_threads = 1
    # sess = ort.InferenceSession("../models/vit_b_16.onnx", sess_options=opt)
    sess = altius_py.InferenceSession("../models/vit_b_16.onnx", True)

    inputs = {"x": input}
    start = time.time()
    output = sess.run(None, inputs)[0].reshape(1000)
    print(f"elapsed: {time.time() - start}")
    output = np.argsort(output)[::-1][:5]
    output = [labels[i].strip() for i in output]
    print(f"top5: {output}")


if __name__ == "__main__":
    main()
