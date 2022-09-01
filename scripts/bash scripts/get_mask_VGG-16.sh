#!/bin/sh

(
cd ../
name=$1
label=$2

CUDA_VISIBLE_DEVICES=0 python masked_print.py \
  --name=${name:="162_166.JPEG"} \
  --label=${label:="162"}
)