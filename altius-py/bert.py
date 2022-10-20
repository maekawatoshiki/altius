# python -m transformers.onnx --model=bert-base-cased --feature=masked-lm ./a

import time
from transformers import AutoTokenizer, BertTokenizer
import onnxruntime as ort
import numpy as np
import altius_py
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased", mask_token="[MASK]")
session = ort.InferenceSession("./a/simple.onnx")
session = altius_py.InferenceSession(
    "../models/bert.onnx", intra_op_num_threads=8, enable_profile=True
)

# msg = "Paris is the [MASK] city of France"
# msg = "Deep [MASK] network has been widely used"
msg = "We usually use a [MASK] to input characters to a computer"
# msg = "The number [MASK] is famous as the ultimate answer of everything"
mask_pos = msg.split().index("[MASK]") + 1
print(f"Masked sentence (up to 20 tokens): {msg}")

inputs = tokenizer(msg, return_tensors="np")
for name in ["input_ids", "attention_mask", "token_type_ids"]:
    input = np.zeros((1, 20), dtype=np.int64)
    input[0, : inputs[name].shape[1]] = inputs[name]
    inputs[name] = input

repeat = 10  # TIPS: First run is usually slow.
for _ in range(repeat):
    start = time.time()
    outputs = session.run(None, dict(inputs))
    end = time.time()
    print(f"Inference time: {end - start}")

    ids = np.argsort(-outputs[0][0, mask_pos])[:5]
    for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids.tolist())):
        print(f"Top{i+1}: {msg.replace('[MASK]', tok.upper())}")
