#!/bin/sh -eux

DIR=TinyStories-33M

optimum-cli export onnx \
  -m 'roneneldan/TinyStories-33M' \
  --opset 13 \
  --task causal-lm \
  $DIR

onnxsim \
  $DIR/decoder_model.onnx $DIR/decoder_model.onnxsim.onnx \
  --overwrite-input-shape input_ids:1,100 attention_mask:1,100

