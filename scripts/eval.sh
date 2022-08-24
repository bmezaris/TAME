#!/bin/sh

(
#overwrite default MODEL, LAYERS, VER, ARR, MLR with wrapper job script
if [ -z ${MODEL+x} ]; then MODEL="vgg16"; fi
if [ -z ${LAYERS+x} ]; then LAYERS="features.15 features.22 features.29"; fi
if [ -z ${VER+x} ]; then VER="V5"; fi
if [ -z ${ARR+x} ]; then ARR="1-1"; fi
if [ -z ${ARCH+x} ]; then ARCH=""; fi
if [ -z ${EPOCHS+x} ]; then EPOCHS=8; fi
# --test-list="Test_2000.txt" \
CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
	--test-list="Test_2000.txt" \
	--num-workers=4 \
  --model=$MODEL \
  --arch=$ARCH \
	--layers="$LAYERS" \
	--version=$VER \
	--arrangement=$ARR \
	--start-epoch=1 \
	--end-epoch=$EPOCHS

exit 0
)