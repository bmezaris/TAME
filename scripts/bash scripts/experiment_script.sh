#!/bin/sh



(
cd ../


CUDA_VISIBLE_DEVICES=0 python train_script.py \
	--img-dir=$IMGDIR \
	--restore-from=$RESTORE \
	--train-list=${TRAIN:=VGG16_train.txt} \
  --num-workers=4 \
	--model=${MODEL:=vgg16} \
  --version="${VERSION:="TAME"}" \
	--layers="${LAYERS:="features.16 features.23 features.30"}" \
  --wd=${WD:=5e-4} \
	--max-lr=${MLR:=1e-2} \
	--epoch=${EPOCHS:=8} \
	--batch-size=${BSIZE:=32}


CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
  --restore-from=$RESTORE \
	--test-list=${TEST:="Evaluation_2000.txt"} \
	--num-workers=4 \
  --model=${MODEL:="vgg16"} \
  --version=${VERSION:="TAME"} \
	--layers="${LAYERS:="features.16 features.23 features.30"}" \
	--start-epoch=1 \
	--end-epoch=${EPOCHS:=32}


exit 0
)
