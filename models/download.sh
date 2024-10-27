#!/bin/bash -eux

if [ "${1:-}" = "CI" ]; then
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/mnist-8.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/mobilenetv3.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/cat.png
else
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/mnist-8.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/mobilenetv3.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/deeplab_mobilenetv3.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/fcn-resnet50.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/yolov5s.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/realesrgan_256x256.onnx
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/cat.png
    wget https://pub-edba5feea2c145019e8be2a71dbeea81.r2.dev/dog.jpg
fi

