import os
import time

from urllib.request import urlopen
from PIL import Image

import onnx
import onnxsim

import torch
import timm
import onnxruntime as ort
import altius_py as alt


def main():
    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )

    model = timm.create_model("fastvit_s12.apple_in1k", pretrained=True)
    model = model.eval()

    if not os.path.exists("fastvit.onnx"):
        torch.onnx.export(
            model,
            torch.randn(1, 3, 256, 256),
            "fastvit.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
        )
        _model, check = onnxsim.simplify(onnx.load("fastvit.onnx"))
        assert check, "Failed to simplify model"
        onnx.save(_model, "fastvit.onnx")

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # For altius
    os.environ["GOMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["GOMP_CPU_AFFINITY"] = "0-7"

    # ort_sess = ort.InferenceSession("fastvit.onnx", providers=["CPUExecutionProvider"])
    alt_sess = alt.InferenceSession(
        "fastvit.onnx", backend="cpu", intra_op_num_threads=8
    )

    with open("../../models/imagenet_classes.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
        class_idx_to_label = {i: class_names[i] for i in range(len(class_names))}

    output = alt_sess.run(
        None,
        {"input": transforms(img).unsqueeze(0).numpy()},
    )[0]

    top5_probabilities, top5_class_indices = torch.topk(
        torch.tensor(output).softmax(dim=1) * 100, k=5
    )
    print(f"top 5 probs: {top5_probabilities}")
    print(
        f"top 5 labels: {[class_idx_to_label[idx] for idx in top5_class_indices.squeeze(0).tolist()]}"
    )


if __name__ == "__main__":
    main()
