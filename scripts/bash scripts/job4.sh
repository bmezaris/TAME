#!/bin/sh

(
. pc_info.sh

LAYERS="features.30"
#ARR=""

#MODEL="resnet50"
#LAYERS="layer2 layer3 layer4"
VER="V5"
MLR="1e-3"
#WD="1e-4"
#OPTIM="AdamW"
ARCH="_1layer_f30"
#BN="true"
EPOCHS=8

cd ../
. experiment_script.sh
exit 0
)
exit 0
