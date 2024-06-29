#!/bin/bash -eux

export LANG=C
export LC_ALL=C

if [ -d env ]; then
    source env/bin/activate
fi

if ! command -v gdown > /dev/null 2>&1; then
    python3 -m venv env
    source env/bin/activate
    pip install gdown
fi

down() {
    if [ ! -f "$2" ]; then
        gdown "$1" -O "$2"
    fi
}

if [ "${1:-}" = "CI" ]; then
    down 1BiXmAGt_SZdZ1OuSv3ntT6Ad5h_q7oJ4 ./mnist-8.onnx
    down 1cZtpzvERn-QXDjfbPYY_cu3RlxxQJOVP ./mobilenetv3.onnx
    down 1PkSkHolMuM8_Eefj4Nu0LDSF_xqezgsT ./cat.png
else
    down 1E2RxWfxufLNB_cXm30RdjXt4YVbFzQrs ./bert.onnx
    down 1YP8wJyOhR0vSaeasn-z1WbkXBX8TzUER ./realesrgan_256x256.onnx
    down 1BiXmAGt_SZdZ1OuSv3ntT6Ad5h_q7oJ4 ./mnist-8.onnx
    down 1cZtpzvERn-QXDjfbPYY_cu3RlxxQJOVP ./mobilenetv3.onnx
    down 1PkSkHolMuM8_Eefj4Nu0LDSF_xqezgsT ./cat.png
    down 1QPbKB7KjJxIXe3Zv3Q5HqrdwQOJWMTLt ./dog.jpg
    down 1KsIguzhvffIKFYIDhAMFWxU_cii9DOJT ./deeplab_mobilenetv3.onnx
    down 1HZ__4-EqloRWwXZJrMlCZvyGZteY64WO ./fcn-resnet50.onnx
    down 129ns91SK-LEv6kWy5hNA86uZhMJe6FDl ./yolov5s.onnx
fi

