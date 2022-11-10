#!/bin/bash -eux

download() {
  ID=${1?}
  OUT=${2?}

  if [ -e $OUT ]; then
    return
  fi

  CONFIRM=$( \
    wget \
      --quiet \
      --save-cookies /tmp/cookies.txt \
      --keep-session-cookies \
      --no-check-certificate \
      "https://drive.google.com/uc?export=download&id=$ID" \
      -O- | \
    sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$ID" -O $OUT
  rm -f /tmp/cookies.txt
}

download 1K6m7WxoE3noMTbNPyhyxLLNlXsdN1Sq6 ./vit_b_16.onnx
download 1E2RxWfxufLNB_cXm30RdjXt4YVbFzQrs ./bert.onnx
download 1YP8wJyOhR0vSaeasn-z1WbkXBX8TzUER ./realesrgan_256x256.onnx
download 1GgqkMVXL0T2ssZ-JSrMhCCJvAmpIgk8Q ./gpt2.onnx
download 1BiXmAGt_SZdZ1OuSv3ntT6Ad5h_q7oJ4 ./mnist-8.onnx
download 1cZtpzvERn-QXDjfbPYY_cu3RlxxQJOVP ./mobilenetv3.onnx
