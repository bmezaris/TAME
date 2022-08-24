#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python Evaluation_L_CAM.py \
	--img_dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_val' \
	--snapshot_dir='/ssd/ntrougkas/L-CAM/snapshots/ResNet50_L_CAM_Img' \
	--num_workers=4 \
	--arch='ResNet50_L_CAM_Img' \
	--model='resnet50' \
	--layers='layer4' \
	--percentage="$1" \
)