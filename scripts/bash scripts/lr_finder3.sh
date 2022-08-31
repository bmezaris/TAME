#!/bin/sh

(
. pc_info.sh

ARCH="_adamw99"
#LAYERS="features.15 features.22 features.29"
VER="V4.1.3"
#BN="true"
WD="1e-4"
OPTIM="AdamW"
B2="0.99"

MODEL="resnet50"
LAYERS="layer2 layer3 layer4"
#VER="V4.1.3"

cd ../

. lr_finder.sh

exit 0
)