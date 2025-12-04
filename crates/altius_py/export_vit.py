import torch
import torchvision, torchvision

import onnx, onnx.checker, onnx.shape_inference
import onnxslim
from torchvision.models import ViT_B_16_Weights

model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
model.eval()

path = "../../models/vit_b_16.onnx"
torch.onnx.export(model, torch.randn(1, 3, 224, 224), path, opset_version=14)

simplified_model = onnxslim.slim(path)
onnx.save(simplified_model, path)
