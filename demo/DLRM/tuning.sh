#!/bin/bash
AR=(1 2 4 8 16 32 64 128 256 512 1024 2048)
length=${#AR[@]}
tuning="tuning"
mkdir -p "./${tuning}"
pth_twm="${PWD}/../../"

export TVM_HOME=$pth_twm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

for (( i = 0; i < length; i++ )); do
  fname="${tuning}/log_${AR[$i]}.json"
  python ./tune_dlrm.py --output-log="${fname}" $1 --batch-size=${AR[$i]}
done