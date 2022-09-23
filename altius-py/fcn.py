import time
import os
import random
import logging

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

import onnxruntime as ort
import altius_py


def main():
    logging.basicConfig(level=logging.INFO)

    image = Image.open("../models/cat.png")

    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    input = np.ascontiguousarray((preprocess(image).unsqueeze(0)))

    # sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = 8
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # sess = ort.InferenceSession("fcn.onnx", sess_options=sess_options)
    sess = altius_py.InferenceSession("../models/fcn-resnet50.onnx")

    inputs = {"input.1": input}
    import time

    start = time.time()
    output = sess.run(None, inputs)[0]
    print(f"Inference elapsed: {time.time() - start}")

    prediction = torch.tensor(output)
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["cat"]]
    masked_img = to_pil_image(mask)
    masked_img.show()
    masked_img.save("masked_cat.png")


if __name__ == "__main__":
    main()
