import time

import coremltools
import torch
import torchvision

from PIL import Image
import numpy as np
from torchvision import transforms


labels = open("../../models/imagenet_classes.txt").readlines()
image = Image.open("../../models/cat.png")

model = torchvision.models.mobilenet_v3_large(pretrained=True)
model.eval()
model = torch.jit.trace(model, torch.zeros(1, 3, 224, 224))

coreml_model = coremltools.convert(
    model,
    inputs=[coremltools.TensorType(name="input_1", shape=(1, 3, 224, 224))],
    outputs=[coremltools.TensorType(name="output_1")],
)

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

for i in range(100):
    start = time.time()
    pred = coreml_model.predict({"input_1": input})["output_1"][0]
    print(f"elapsed: {(time.time() - start) * 1000:.2f}ms")
output = np.argsort(pred)[::-1][:5]
output = [labels[i].strip() for i in output]
print(f"top5: {output}")
