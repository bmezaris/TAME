#!/bin/sh

(
. pc_info.sh

ARCH="_2layer"
LAYERS="features.22 features.29"
VER="V5"
#BN="true"
#WD="1e-2"
#OPTIM="AdamW"
MODE="all"


#MODEL="resnet50"
#LAYERS="layer2 layer3 layer4"
#VER="V4.1.3"

cd ../

. lr_finder.sh

exit 0
)