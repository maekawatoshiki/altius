#!/bin/bash -eux

export LANG=C
export LC_ALL=C

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

if [ "${1:-}" = "CI" ]; then
  download 1BiXmAGt_SZdZ1OuSv3ntT6Ad5h_q7oJ4 ./mnist-8.onnx
  download 1cZtpzvERn-QXDjfbPYY_cu3RlxxQJOVP ./mobilenetv3.onnx
  download 1PkSkHolMuM8_Eefj4Nu0LDSF_xqezgsT ./cat.png
else
  download 1E2RxWfxufLNB_cXm30RdjXt4YVbFzQrs ./bert.onnx
  download 1YP8wJyOhR0vSaeasn-z1WbkXBX8TzUER ./realesrgan_256x256.onnx
  download 1BiXmAGt_SZdZ1OuSv3ntT6Ad5h_q7oJ4 ./mnist-8.onnx
  download 1cZtpzvERn-QXDjfbPYY_cu3RlxxQJOVP ./mobilenetv3.onnx
  download 1PkSkHolMuM8_Eefj4Nu0LDSF_xqezgsT ./cat.png
  download 1QPbKB7KjJxIXe3Zv3Q5HqrdwQOJWMTLt ./dog.jpg
  download 1KsIguzhvffIKFYIDhAMFWxU_cii9DOJT ./deeplab_mobilenetv3.onnx
  download 1HZ__4-EqloRWwXZJrMlCZvyGZteY64WO ./fcn-resnet50.onnx
  download 129ns91SK-LEv6kWy5hNA86uZhMJe6FDl ./yolov5s.onnx
fi

