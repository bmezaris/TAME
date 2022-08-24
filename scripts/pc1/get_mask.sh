#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python masked_print.py \
	--img_dir='/ssd/ntrougkas/L-CAM/images/' \
	--snapshot_dir='/ssd/ntrougkas/L-CAM/snapshots/' \
	--arch='' \
  --model='resnet50' \
	--layers="layer2 layer3 layer4" \
	--version='V5' \
)