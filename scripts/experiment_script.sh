#!/bin/sh

(

trap "exit 0" INT TERM
trap "kill 0" EXIT

#overwrite default MODEL, LAYERS, VER, ARR, MLR with wrapper job script
if [ -z ${MODEL+x} ]; then MODEL="vgg16"; fi
if [ -z ${BN+x} ]; then BN="false"; fi
if [ -z ${LAYERS+x} ]; then LAYERS="features.15 features.22 features.29"; fi
if [ -z ${OPTIM+x} ]; then OPTIM="SGD"; fi
if [ -z ${WD+x} ]; then WD="5e-4"; fi
if [ -z ${B2+x} ]; then B2="0.999"; fi
if [ -z ${VER+x} ]; then VER="V3.2"; fi
if [ -z ${ARR+x} ]; then ARR="1-1"; fi
if [ -z ${MLR+x} ]; then MLR="3e-3"; fi
if [ -z ${ARCH+x} ]; then ARCH=""; fi
if [ -z ${EPOCHS+x} ]; then EPOCHS=8; fi
if [ -z ${BSIZE+x} ]; then BSIZE=8; fi
if [ -z ${SCHED+x} ]; then SCHED=onecycle; fi
if [ -z ${TRAIN+x} ]; then TRAIN="VGG16_train.txt"; fi



GAMMA="0.95"

(
cd ../


(
CUDA_VISIBLE_DEVICES=0 python train_script.py \
	--img-dir=$IMGDIR \
	--num-workers=4 \
	--train-list=$TRAIN \
	--model=$MODEL \
	--freeze-bn=$BN \
	--arch=$ARCH \
	--layers="$LAYERS" \
  --optim=$OPTIM \
  --wd=$WD \
  --b2=$B2 \
	--version=$VER \
	--arrangement=$ARR \
	--max-lr=$MLR \
	--gamma=$GAMMA \
	--epoch=$EPOCHS \
	--batch-size=$BSIZE \
	--schedule=$SCHED

exit 0
)

(
CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
	--num-workers=4 \
  --model=$MODEL \
  --arch=$ARCH \
	--layers="$LAYERS" \
	--version=$VER \
	--arrangement=$ARR \
	--start-epoch=1 \
	--end-epoch=$EPOCHS

exit 0
)

exit 0

)
exit 0
