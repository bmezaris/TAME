#!/bin/sh

(
. pc_info.sh

ARR="hyper: 1, 2, 0.01703"

cd ../
. hyper_eval.sh
exit 0
)
