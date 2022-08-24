#!/bin/sh

(
trap "exit" INT TERM
trap "kill 0" EXIT

MODEL="vgg16"
LAYERS="features.15 features.22 features.29"

VER="V4.1.3"
ARR="1-8"
MLR="1e-2"

EPOCHS="8"
BSIZE="32"

(
cd ../../

tensorboard dev upload \
  --logdir "snapshots/data/logs/${MODEL}_${VER}_(${ARR})" \
  --name "${MODEL}_${VER}_(${ARR})" \
  &
) &> /dev/null

(
cd ../

CUDA_VISIBLE_DEVICES=0 python t_train_script.py \
	--img-dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train' \
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

exit
)