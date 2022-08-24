#!/bin/sh

(
if [ -z ${MODEL+x} ]; then MODEL="vgg16"; fi
if [ -z ${BN+x} ]; then BN="False"; fi
if [ -z ${LAYERS+x} ]; then LAYERS="features.15 features.22 features.29"; fi
if [ -z ${OPTIM+x} ]; then OPTIM="SGD"; fi
if [ -z ${WD+x} ]; then WD="5e-4"; fi
if [ -z ${B2+x} ]; then B2="0.999"; fi
if [ -z ${VER+x} ]; then VER="V3.2"; fi
if [ -z ${ARR+x} ]; then ARR="1-1"; fi
if [ -z ${ARCH+x} ]; then ARCH=""; fi
if [ -z ${BSIZE+x} ]; then BSIZE=8; fi
if [ -z ${MODE+x} ]; then MODE="tot"; fi
if [ -z ${STOP+x} ]; then STOP="True"; fi



CUDA_VISIBLE_DEVICES=0 python lr_finder.py \
	--img-dir=$IMGDIR \
	--num-workers=4 \
	--arch=$ARCH \
	--model=$MODEL \
  --freeze-bn=$BN \
	--layers="$LAYERS" \
	--optim=$OPTIM \
	--wd=$WD \
	--b2=$B2 \
	--version=$VER \
	--arrangement="$ARR" \
	--batch-size=$BSIZE \
	--mode=$MODE \
	--early-stop=$STOP

exit 0
)