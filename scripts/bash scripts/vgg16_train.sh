#!/bin/sh

(
. pc_info.sh

MODEL="resnet50"
LAYERS="layer2 layer3 layer4"

#LAYERS="features.30"
VER="V1"
MLR="1e-2"
ARCH="_hyper_superhilr"
ARR="hyper"
#BN="false"
EPOCHS=8

cd ../
. experiment_script.sh
exit 0
)