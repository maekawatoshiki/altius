#!/bin/bash -eux

EXPORTER_VENV=.exporter.venv

if [ ! -d ${EXPORTER_VENV} ]; then
  python3 -m venv ${EXPORTER_VENV}
  source ${EXPORTER_VENV}/bin/activate
  pip install -U pip
  pip install onnx onnxruntime onnxsim optimum==1.16.2
fi

source ${EXPORTER_VENV}/bin/activate

DIR=bert-onnx

python -m optimum.exporters.onnx --model "bert-base-uncased" --task fill-mask --opset 14 ${DIR}

ONNXSIM_FIXED_POINT_ITERS=1000 \
onnxsim ./${DIR}/model.onnx ./${DIR}/model.onnx --overwrite-input-shape input_ids:1,100 attention_mask:1,100 token_type_ids:1,100

printf "\e[1;32mExported in ${DIR}\e[0m\n"
