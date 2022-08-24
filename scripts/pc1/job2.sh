#!/bin/sh

(
. pc_info.sh

#LAYERS="features.15 features.22 features.29"
#ARR=""

#MODEL="resnet50"
#LAYERS="layer3 layer4"
VER="V3.2"
MLR="1e-2"
#WD="1e-3"
ARCH="_big_lr"
#BN="true"
BSIZE=32
EPOCHS=8

cd ../
. experiment_script.sh
exit 0
)
