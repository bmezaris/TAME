#!/bin/sh

(
. pc_info.sh

#LAYERS="features.15 features.22 features.29"
#ARR=""

MODEL="resnet50"
LAYERS="layer2 layer3 layer4"
VER="V5"
ARCH=""
#ARR="hyper"
EPOCHS=8

cd ../
. eval.sh
exit 0
)
