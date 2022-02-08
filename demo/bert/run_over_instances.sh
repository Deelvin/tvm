#!/bin/bash
instances=(1 2 4 8 10 12 14 16)
threads=(1 2 4 8 10 12 14 16)
length=${#instances[@]}
length_thr=${#threads[@]}
tuning="tuning"
mkdir -p "./${tuning}"
pth_twm="${PWD}/../../"

export TVM_HOME=$pth_twm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

for (( i = 0; i < length; i++ )); do
  for (( j = 0; j < length_thr; j++ )); do
    fname="${tuning}/log_i${instances[$i]}_t${threads[$j]}.txt"
    # python ./tune_dlrm.py --output-log="${fname}" $1 --batch-size=${AR[$i]}
    echo ${fname} ${instances[$i]} ${threads[$j]}
    python ./multi_instance_BERT.py --model /home/ubuntu/dev/tvm/demo/bert/out/saved_model_3000_b100.tar.so --num-instances ${instances[$i]} --num-threads ${threads[$j]} > ${fname}
  done
done
