import os
import matplotlib.pyplot as plt

import torch.nn as nn
import onnxsim
import onnx
import torch
import onnxruntime as ort
import altius_py

W = 320 * 3
H = 240 * 3
XMIN = -2.4
XMAX = 1.2
YMIN = -1.2
YMAX = 1.2


class Mandelbrot(nn.Module):
    def forward(self, k, zx, zy):
        w = W
        h = H
        x = torch.linspace(XMIN, XMAX, W, dtype=torch.float32)
        y = torch.linspace(YMIN, YMAX, H, dtype=torch.float32)
        cx, cy = torch.meshgrid([x, y])
        cx = cx.to(torch.float32)
        cy = cy.to(torch.float32)

        zx2 = zx**2
        zy2 = zy**2
        inf = (zx2 + zy2) > 4
        max = torch.max(k)
        k[inf] = max + 1
        zxn = zx2 - zy2 + cx
        zyn = 2 * zx * zy + cy
        return k, zxn, zyn


if __name__ == "__main__":
    model = Mandelbrot()

    zx = torch.zeros(W * H, dtype=torch.float32).reshape(W, H)
    zy = torch.zeros(W * H, dtype=torch.float32).reshape(W, H)
    k = torch.zeros(W * H, dtype=torch.float32).reshape(W, H)
    path = "/tmp/mandelbrot.onnx"

    if not os.path.exists(path):
        torch.onnx.export(
            model,
            {"k": k, "zx": zx, "zy": zy},
            path,
            input_names=["k", "zx", "zy"],
            opset_version=12,
        )
        simplified, ok = onnxsim.simplify(onnx.load(path))
        assert ok
        onnx.save(simplified, path)

    model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # model = altius_py.InferenceSession(path)

    k = k.numpy()
    zx = zx.numpy()
    zy = zy.numpy()

    for i in range(100):
        k, zxn, zyn = model.run(None, {"k": k, "zx": zx, "zy": zy})
        zx = zxn
        zy = zyn

    mandelbrot = k.T

    plt.figure(figsize=(3.200, 2.400), dpi=1000)
    img = plt.imshow(mandelbrot)
    img.set_cmap("hot")
    plt.axis("off")
    # plt.savefig("mandel.png", dpi=100)
    plt.show()
