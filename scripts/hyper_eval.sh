#!/bin/sh

(
#overwrite default MODEL, LAYERS, VER, ARR, MLR with wrapper job script
if [ -z ${MODEL+x} ]; then MODEL="resnet50"; fi
if [ -z ${LAYERS+x} ]; then LAYERS="layer2 layer3 layer4"; fi
if [ -z ${VER+x} ]; then VER="V5"; fi
if [ -z ${ARR+x} ]; then ARR="hyper: 1.5, 2, 0.01"; fi
if [ -z ${ARCH+x} ]; then ARCH="_hyper_hilr"; fi
if [ -z ${EPOCHS+x} ]; then EPOCHS=8; fi

CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
	--num-workers=4 \
  --model=$MODEL \
  --arch=$ARCH \
	--layers="$LAYERS" \
	--version=$VER \
	--arrangement="$ARR" \
	--start-epoch=1 \
	--end-epoch=$EPOCHS

exit 0
)