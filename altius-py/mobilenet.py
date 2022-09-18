import altius_py
import numpy as np
from PIL import Image
from torchvision import transforms
import os, random
from matplotlib import pyplot as plt


def main():
    labels = open("../models/imagenet_classes.txt").readlines()
    image = Image.open("../models/cat.png")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input = preprocess(image)
    input = input.unsqueeze(0).numpy()

    sess = altius_py.InferenceSession("../models/mobilenetv3.onnx")

    inputs = {"input": input}
    output = sess.run(None, inputs)[0][0]
    output = np.argsort(output)[::-1][:5]
    output = [labels[i].strip() for i in output]
    print(f"top5: {output}")


if __name__ == "__main__":
    main()
