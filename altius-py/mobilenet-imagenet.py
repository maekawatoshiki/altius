import altius_py
import numpy as np
from PIL import Image
from torchvision import transforms
import os, random
from matplotlib import pyplot as plt
import onnxruntime as ort
from torchvision.models import mobilenetv3

import os
def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)


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

    # model = mobilenetv3.mobilenet_v3_large(pretrained=True)
    # import torch
    # import time
    # for i in range(10):
    #     start = time.time()
    #     t = model(torch.tensor(input))
    #     end = time.time()
    #     print(end - start)

    sess = altius_py.InferenceSession("../models/mobilenetv3.onnx")

    import os

    # traverse root directory, and list directories as dirs and files as files
    # images = []
    # for img_path in list_files("/home/uint/work/aoha/dataset_imagenet/validation/", "JPEG"):
    #     image = Image.open(img_path)
    #     if image.mode != 'RGB':
    #         continue
    #     preprocess = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     )
    #     input = preprocess(image)
    #     input = input.unsqueeze(0).numpy()
    #     images.append(input)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # suppose the files contains th 16k file names
    files = list_files("/home/uint/work/aoha/dataset_imagenet/validation/", "JPEG")
    files = files[:100]
    future_to_file = {}
    images = []

    def process_img(file):
        # print(file)
        image = Image.open(file)
        # if image.mode != 'RGB':
        #     continue
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

        inputs = {"input": input}
        output = sess.run(None, inputs)[0][0]
        output = np.argsort(output)[::-1][:5]
        output = [labels[i].strip() for i in output]

        return output

    with ThreadPoolExecutor(max_workers=2) as executor:
        for file in files:
            future = executor.submit(process_img, file)
            future_to_file[future] = file
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            output = future.result()
            # images.append(img_read)
            print(f"top5: {output}")

    # for input in images:
    #     inputs = {"input": input}
    #     output = sess.run(None, inputs)[0][0]
    #     output = np.argsort(output)[::-1][:5]
    #     output = [labels[i].strip() for i in output]
    #     print(f"top5: {output}")




    # sess_altius = altius_py.InferenceSession("../models/mobilenetv3.onnx")
    # sess_ort = ort.InferenceSession("../models/mobilenetv3.onnx")
    # 
    # for i in range(2):
    #     for sess in [sess_altius, sess_ort]:
    #         inputs = {"input": input}
    #         start = time.time()
    #         output = sess.run(None, inputs)[0][0]
    #         end = time.time()
    #         print(end-start)
    #         output = np.argsort(output)[::-1][:5]
    #         output = [labels[i].strip() for i in output]
    #         print(f"top5: {output}")


if __name__ == "__main__":
    main()
