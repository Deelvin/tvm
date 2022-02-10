#!/bin/bash
instances=(1 2 4 8 10 12 14 16)
threads=(1 2 4 8 10 12 14 16)
length=${#instances[@]}
length_thr=${#threads[@]}
tuning="tuning"
mkdir -p "./${tuning}"
pth_tvm="${PWD}/../../"
# procs=$(nproc)
# procs_2=$((procs/2))

export TVM_HOME=$pth_tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export OMP_NUM_THREADS=8

if [ ! -d "./out" ] 
then
    mkdir ./out
fi

python ./tune_bert.py --output-log ./test_log.json --output-folder ./out
python compile.py --tuning-log-file ./test_log.json --output-name ./out/test_bert --batch-size 1 --onnx-model ../inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx
for (( i = 0; i < length; i++ )); do
  for (( j = 0; j < length_thr; j++ )); do
    fname="${tuning}/bert_log_i${instances[$i]}_t${threads[$j]}.txt"
    echo ${fname} ${instances[$i]} ${threads[$j]}
    python ./multi_instance_BERT.py --model ./out/test_bert_b1.so --num-instances ${instances[$i]} --num-threads ${threads[$j]} > ${fname}
  done
done
