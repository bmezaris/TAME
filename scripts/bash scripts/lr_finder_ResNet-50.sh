#!/bin/sh

(
. pc_info.sh

export MODEL=resnet50
export LAYERS="layer2 layer3 layer4"
export TRAIN=ResNet50_train.txt

export EPOCHS=8

# export BSIZE=8
# export VERSION="TAME"
# export WD=
# export LAYERS=
# export RESTORE=
# export TEST="Validation_2000.txt"


. lr_finder.sh

exit 0
)
