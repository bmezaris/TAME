#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python cyclic_train_script.py \
	--img-dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train' \
	--snapshot-dir='/ssd/ntrougkas/L-CAM/snapshots/' \
	--num-workers=4 \
	--arch='VGG16_L_CAM_Img_C' \
	--model='vgg16' \
	--layers='features.15 features.22 features.29' \
	--version='V3.1' \
	--arrangement='1-1' \
	--min-lr=5e-6 \
	--max-lr=5e-3 \
	--epoch=8 \
	--batch-size=16
)