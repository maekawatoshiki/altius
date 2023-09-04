import time
import os
import random
import logging
from itertools import cycle

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision import transforms
from matplotlib import colors as mcolors
from PIL import Image

import onnxruntime as ort
import altius_py


def main():
    logging.basicConfig(level=logging.INFO)

    path = "../../models/cat.png"
    image = Image.open(path).resize((520, 520))

    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    input = np.ascontiguousarray((preprocess(image).unsqueeze(0)))

    # sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = 1
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # sess = ort.InferenceSession(
    #     "../../models/fcn-resnet50.onnx", sess_options=sess_options
    # )
    sess = altius_py.InferenceSession("../../models/fcn-resnet50.onnx")

    inputs = {"input.1": input}

    start = time.time()
    output = sess.run(None, inputs)[0]
    print(f"Inference elapsed: {time.time() - start}")

    prediction = torch.tensor(output)
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    colors = cycle(mcolors.BASE_COLORS.values())
    color_like = lambda input: [torch.full_like(input, c) for c in next(colors)]

    for klass, idx in class_to_idx.items():
        if klass == "__background__":
            continue

        mask = normalized_masks[0, idx]
        if torch.max(mask) < 0.2:
            # No objects of this class
            continue

        mask_img = to_pil_image(
            torch.stack(color_like(mask) + [mask * 0.5]),
            mode="RGBA",
        )
        image = Image.alpha_composite(image.convert("RGBA"), mask_img)

    image.save("masked.png")
    image.show()


if __name__ == "__main__":
    main()
