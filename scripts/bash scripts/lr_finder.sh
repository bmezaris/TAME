#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python lr_finder.py \
	--img-dir=$IMGDIR \
	--num-workers=4 \
  --train-list=${TRAIN:=VGG16_train.txt} \
	--model=${MODEL:=vgg16} \
	--layers="${LAYERS:="features.16 features.23 features.30"}" \
	--wd=${WD:=5e-4} \
	--version="${VERSION:="TAME"}" \
	--batch-size=${BSIZE:=32} \

exit 0
)