#!/bin/sh

(
. pc_info.sh

ARCH="_2layer"
#LAYERS="features.15 features.22 features.29"
VER="V5"
BN="False"
#WD="1e-6"
#OPTIM="AdamW"

MODEL="resnet50"
LAYERS="layer3 layer4"
#VER="V4.1.3"

cd ../

. lr_finder.sh

exit 0
)