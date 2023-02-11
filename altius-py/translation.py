# python -m optimum.exporters.onnx --model "staka/fugumt-en-ja" --for-ort fugu

import time
import logging
import os
import sys

from transformers import pipeline
from transformers import MarianTokenizer
import onnxruntime as ort
import numpy as np
import altius_py
import torch
from torch.nn import functional as F


def translate_baseline(text_en):
    fugu_translator = pipeline("translation", model="staka/fugumt-en-ja", device="cpu")
    result = fugu_translator(text_en)[0]["translation_text"]
    return result


def translate_onnx(text_en):
    tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    encoder = ort.InferenceSession(
        "./fugu/encoder_model2.onnx", providers=["CPUExecutionProvider"]
    )
    decoder = ort.InferenceSession(
        "./fugu/decoder_model2.onnx", providers=["CPUExecutionProvider"]
    )

    max_tokens = 100
    text = text_en
    text += "</s>"

    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=False,
        add_special_tokens=False,
    )
    len_ = inputs["input_ids"].shape[1]

    assert len_ < max_tokens

    if len_ >= max_tokens:
        raise Exception("Too long")

    for name in ["input_ids", "attention_mask"]:
        input = np.zeros((1, max_tokens), dtype=np.int64)
        input[0, : inputs[name].shape[1]] = inputs[name]
        inputs[name] = input

    last_hidden_state = encoder.run(["last_hidden_state"], dict(inputs))[0]

    translated_text = "<pad>"
    for i in range(100):
        decoder_text = tokenizer(
            translated_text,
            return_tensors="np",
            padding=False,
            text_target="ja",
            add_special_tokens=False,
        )
        len_ = decoder_text["input_ids"].shape[1]

        for name in ["input_ids", "attention_mask"]:
            input = np.zeros((1, max_tokens), dtype=np.int64)
            input[0, : decoder_text[name].shape[1]] = decoder_text[name]
            decoder_text[name] = input

        outputs = decoder.run(
            ["logits", "encoder_last_hidden_state"],
            {
                "encoder_attention_mask": inputs["attention_mask"],
                "input_ids": decoder_text["input_ids"].reshape(1, -1),
                "encoder_hidden_states": last_hidden_state,
            },
        )

        if i >= len_:
            break

        next_token_logits = outputs[0][:, i, :32000]

        probs = F.softmax(torch.tensor(next_token_logits), dim=-1)
        ids = torch.argsort(-probs[0])
        for i in ids:
            if i == 2:
                continue
            if i == tokenizer.pad_token_id:
                print("PAD!")
                continue
            id = i
            break
        resulting_string = tokenizer.decode(
            [id],
            skip_special_tokens=True,  # clean_up_tokenization_spaces=False
        )
        print(resulting_string)
        translated_text += resulting_string

    _, translated_text = translated_text.split("<pad>")

    return translated_text


def main():
    text = "Attention is all you need."

    baseline_result = translate_baseline(text)
    onnx_result = translate_onnx(text)
    print(f"baseline: {baseline_result}")
    print(f"onnx: {onnx_result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
