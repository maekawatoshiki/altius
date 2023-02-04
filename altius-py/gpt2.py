# python -m transformers.onnx --model=gpt2 --feature=causal-lm ./a

import time
import logging
import os
import sys

from transformers import AutoTokenizer, BertTokenizer, top_k_top_p_filtering
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import onnxruntime as ort
import numpy as np
import altius_py
import torch
from torch.nn import functional as F


logging.basicConfig(level=logging.INFO)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
sess = altius_py.InferenceSession(
    "../models/gpt2.onnx", intra_op_num_threads=16, enable_profile=True
)
# sess = ort.InferenceSession("../models/gpt2.onnx", providers=["CPUExecutionProvider"])

torch.manual_seed(42)

max_tokens = 100
text = "Rust is a multi-paradigm, general-purpose programming language. Rust emphasizes performance,"
for _ in range(1000):
    inputs = tokenizer(text, return_tensors="np")
    len = inputs["input_ids"].shape[1]

    if len >= max_tokens:
        break

    for name in ["input_ids", "attention_mask"]:
        input = np.zeros((1, max_tokens), dtype=np.int64)
        input[0, : inputs[name].shape[1]] = inputs[name]
        inputs[name] = input

    outputs = sess.run(None, dict(inputs))

    next_token_logits = outputs[0][:, len - 1, :]

    filtered_next_token_logits = top_k_top_p_filtering(
        torch.tensor(next_token_logits), top_k=50, top_p=1.0
    )
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([torch.tensor(inputs["input_ids"][0, :len]), next_token[0]])
    resulting_string = tokenizer.decode(generated.tolist())
    print(resulting_string)
    text = resulting_string
