#!/bin/bash -eux

EXPORTER_VENV=.exporter.venv

if [ ! -d ${EXPORTER_VENV} ]; then
  python3 -m venv ${EXPORTER_VENV}
  source ${EXPORTER_VENV}/bin/activate
  pip install -U pip
  pip install onnx onnxruntime optimum==1.16.2
fi

source ${EXPORTER_VENV}/bin/activate

DIR=fugumt-en-ja

python -m optimum.exporters.onnx --model "staka/${DIR}" ${DIR}

onnxsim ./${DIR}/encoder_model.onnx ./${DIR}/encoder_model.onnx --overwrite-input-shape input_ids:1,100 attention_mask:1,100
onnxsim ./${DIR}/decoder_model.onnx ./${DIR}/decoder_model.onnx --overwrite-input-shape encoder_attention_mask:1,100 input_ids:1,100 encoder_hidden_states:1,100,512

onnxsim ./${DIR}/decoder_model.onnx ./${DIR}/decoder_model.onnx --unused-output \
  present.0.encoder.key present.1.encoder.key present.2.encoder.key present.3.encoder.key present.4.encoder.key present.5.encoder.key \
  present.0.encoder.value present.1.encoder.value present.2.encoder.value present.3.encoder.value present.4.encoder.value present.5.encoder.value \
  present.0.decoder.key present.1.decoder.key present.2.decoder.key present.3.decoder.key present.4.decoder.key present.5.decoder.key \
  present.0.decoder.value present.1.decoder.value present.2.decoder.value present.3.decoder.value present.4.decoder.value present.5.decoder.value

printf "\e[1;32mExported in ${DIR}\e[0m\n"
