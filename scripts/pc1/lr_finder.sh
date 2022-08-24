#!/bin/sh

(
. pc_info.sh

ARCH="_f30"
LAYERS="features.16 features.23 features.30"
#BN="true"
#WD="5e-4"
MODE="all"
#B2="0.99"
BSIZE=8

#MODEL="resnet50"
#LAYERS="layer2 layer3 layer4"
VER="V5"

cd ../

. lr_finder.sh

exit 0
)
