#!/bin/bash -eux

download() {
  S1=${1?}
  S2=${2?}
  CONFIRM=$( \
    wget \
      --quiet \
      --save-cookies /tmp/cookies.txt \
      --keep-session-cookies \
      --no-check-certificate \
      "https://drive.google.com/uc?export=download&id=$S1" \
      -O- | \
    sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$S1" -O $S2
  rm -f /tmp/cookies.txt
}

download 1K6m7WxoE3noMTbNPyhyxLLNlXsdN1Sq6 ./vit_b_16.onnx
