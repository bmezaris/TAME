#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python Train_L_CAM.py \
	--img-dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train' \
	--snapshot-dir='/ssd/ntrougkas/L-CAM/snapshots/' \
	--train-list='/ssd/ntrougkas/L-CAM/datalist/ILSVRC/VGG16_train.txt' \
	--num-workers=4 \
	--arch='VGG16_L_CAM_Img' \
	--model='vgg16' \
	--layers='features.15 features.22 features.29' \
	--version='V4.1.3' \
	--arrangement='1-1' \
	--lr=0.001 \
	--gamma=0.75 \
	--epoch=8 \
	--batch-size=32
)