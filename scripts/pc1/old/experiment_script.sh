#!/bin/sh

(

trap "exit 1" INT TERM
trap "kill 0" EXIT

if [ -z ${MODEL+x} ]; then MODEL="vgg16"; fi
if [ -z ${LAYERS+x} ]; then LAYERS="features.15 features.22 features.29"; fi
if [ -z ${VER+x} ]; then VER="V4.1.3"; fi
if [ -z ${ARR+x} ]; then ARR="1-1"; fi
if [ -z ${MLR+x} ]; then MLR="3e-3"; fi
if [ -z ${ARCH+x} ]; then ARCH=""; fi


GAMMA="0"
EPOCHS="8"
BSIZE="32"

IMGDIR="/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train"
VALDIR="/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_val"

(
cd ../../

tensorboard dev upload \
  --logdir "snapshots/data/logs/${MODEL}_${VER}_(${ARCH})" \
  --name "${MODEL}_${VER}_(${ARR})" \
  &
) &> /dev/null

(
cd ../

CUDA_VISIBLE_DEVICES=0 python t_train_script.py \
	--img-dir=$IMGDIR \
	--num-workers=4 \
	--model=$MODEL \
	--layers="$LAYERS" \
	--version=$VER \
	--arrangement=$ARR \
	--max-lr=$MLR \
	--gamma=$GAMMA \
	--epoch=$EPOCHS \
	--batch-size=$BSIZE \
)

(
cd ../

CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
	--num-workers=4 \
  --model=$MODEL \
	--layers="$LAYERS" \
	--version=$VER \
	--arrangement=$ARR \
	--start-epoch=0 \
	--end-epoch=8 \
)

exit 0

)