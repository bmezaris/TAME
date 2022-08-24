#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python Train_L_CAM.py \
	--img_dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train' \
	--snapshot_dir='/ssd/ntrougkas/L-CAM/snapshots/ResNet50_L_CAM_Img' \
	--train_list='/ssd/ntrougkas/L-CAM/datalist/ILSVRC/ResNet50_train.txt' \
	--num_workers=4 \
	--arch='ResNet50_L_CAM_Img' \
	--model='resnet50' \
	--layers='layer4' \
	--lr=0.0001 \
	--gamma=0.95 \
	--epoch=26 \
)
