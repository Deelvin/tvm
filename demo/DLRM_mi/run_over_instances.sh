#!/bin/bash
instances=(1 2 4 8 16 20 24 32 48 64)
threads=(1 2 4 8 10 12 14 16 18 20 24 32 48 64)
length=${#instances[@]}
length_thr=${#threads[@]}
tuning="tuning"
mkdir -p "./${tuning}"
pth_twm="${PWD}/../../"

export TVM_HOME=$pth_twm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
python ./compile.py --tuning-log-file ../DLRM/log_b100_thr8.json --output-name dlrm_test --onnx-model ../model/dlrm_s_pytorch_0505.
for (( i = 0; i < length; i++ )); do
  for (( j = 0; j < length_thr; j++ )); do
    fname="${tuning}/log_i${instances[$i]}_t${threads[$j]}.txt"
    echo ${fname} ${instances[$i]} ${threads[$j]}
    python ./multi_instance.py --model ./__prebuilt/dlrm_test_b100.so --num-instances ${instances[$i]} --num-threads ${threads[$j]} > ${fname}
  done
done