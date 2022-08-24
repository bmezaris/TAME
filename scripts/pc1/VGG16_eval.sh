#!/bin/sh

(
cd ../

CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir='/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_val' \
	--snapshot-dir='/ssd/ntrougkas/L-CAM/snapshots/' \
	--test-list='/ssd/ntrougkas/L-CAM/datalist/ILSVRC/Evaluation_2000.txt' \
	--num-workers=4 \
	--arch='VGG16_L_CAM_Img_C' \
  --model='vgg16' \
	--layers='features.15 features.22 features.29' \
	--version='V3.1' \
	--arrangement='1-1' \
	--start-epoch=0 \
	--end-epoch=8
)