import altius_py
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import os, random
from matplotlib import pyplot as plt
import onnxruntime as ort
import onnx
import time

coco_labels = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype="float32")
    image_data /= 255.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def main():
    root = "../models/yolov3/"
    image = Image.open("../models/dog.jpg")
    input = np.ascontiguousarray(preprocess(image))

    model = onnx.load(root + "altius.onnx")
    model_outputs = [x.name for x in model.graph.output]

    sess = altius_py.InferenceSession(root + "altius.onnx")

    inputs = {"input_1": input}

    start = time.time()
    outputs = sess.run(None, inputs)
    print("altius elapsed:", time.time() - start)
    sess_outputs = dict(zip(model_outputs, outputs))

    # Check if altius output is correct compared to onnx runtime output
    # sess = ort.InferenceSession(root + "altius.onnx", providers=["CPUExecutionProvider"])
    # start = time.time()
    # ort_outputs = sess.run(None, inputs)
    # end = time.time() - start
    # print("ort elapsed:", end)
    # assert np.allclose(outputs[0], ort_outputs[0], atol=1e3)
    # assert np.allclose(outputs[1], ort_outputs[1], atol=1e3)
    # assert np.allclose(outputs[2], ort_outputs[2], atol=1e3)

    sess1 = ort.InferenceSession(
        root + "ort_0.onnx", providers=["CPUExecutionProvider"]
    )
    sess2 = ort.InferenceSession(
        root + "ort_1.onnx", providers=["CPUExecutionProvider"]
    )

    image_shape = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(
        1, 2
    )
    sess1_outputs = sess1.run(None, {"image_shape": image_shape})
    sess1_outputs = dict(zip([x.name for x in sess1.get_outputs()], sess1_outputs))

    sess1_outputs.update(sess_outputs)
    sess2_outputs = sess2.run(None, sess1_outputs)
    output_boxes = sess2_outputs[2]
    output_scores = sess2_outputs[1]
    output_indices = sess2_outputs[0]

    print(f"indices: {output_indices.shape}")
    print(f"boxes: {output_boxes.shape}")
    print(f"scores: {output_scores.shape}")

    # Result
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in output_indices:
        # print(idx_)
        out_classes.append(idx_[1])
        out_scores.append(output_scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(output_boxes[idx_1])

    # Make Figure and Axes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    caption = []
    draw_box_p = []
    for i in range(0, len(out_classes)):
        box_xy = out_boxes[i]
        p1_y = box_xy[0]
        p1_x = box_xy[1]
        p2_y = box_xy[2]
        p2_x = box_xy[3]
        draw_box_p.append([p1_x, p1_y, p2_x, p2_y])
        draw = ImageDraw.Draw(image)
        # Draw Box
        draw.rectangle(draw_box_p[i], outline=(255, 0, 0), width=5)

        caption.append(coco_labels[out_classes[i]])
        caption.append("{:.2f}".format(out_scores[i]))
        # Draw Class name and Score
        ax.text(
            p1_x,
            p1_y,
            ": ".join(caption),
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
        )

        caption.clear()

    # Output result image
    img = np.asarray(image)
    ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
