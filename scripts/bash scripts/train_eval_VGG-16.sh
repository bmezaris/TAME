#!/bin/sh

(
. pc_info.sh

export MLR=1e-3
export EPOCHS=8
export BSIZE=32

# export VERSION="TAME"
# export WD=
# export LAYERS=
# export RESTORE=
# export TEST="Validation_2000.txt"

. experiment_script.sh
exit 0
)