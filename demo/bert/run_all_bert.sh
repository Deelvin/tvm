#!/bin/bash
# instances=(1 2 4 8 10 12 14 16)
# threads=(1 2 4 8 10 12 14 16)
# length=${#instances[@]}
# length_thr=${#threads[@]}
tuning="tuning"
mkdir -p "./${tuning}"
pth_tvm="${PWD}/../../"
length_thr=$(nproc)
length_thr=$((length_thr + 1))

procs=$(nproc)
procs_2=$((procs/2))

export TVM_HOME=$pth_tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
# export OMP_NUM_THREADS=8

if [ ! -d "./out" ] 
then
    mkdir ./out
fi

# python ./tune_bert.py --output-log ./test_log.json --output-folder ./out
# python compile.py --tuning-log-file ./test_log.json --output-name ./out/test_bert --batch-size 1 --onnx-model ../inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx

for (( i = 1; i < length_thr; i++ )); do
  threads=$(($(nproc)/i))
  threads=$((threads + 1))
  for (( j = 1; j < threads; j++ )); do
    fname="${tuning}/bert_log_v1_i${i}_t${j}.txt"
    echo ${fname} ${i} ${j}
    python ./multi_instance_BERT.py --model ./out/test_bert_b1.so --num-instances ${i} --num-threads ${j} > ${fname}
  done
done

for (( i = procs_2; i < length_thr; i++ )); do
  for (( j = 2; j < 3; j++ )); do
    fname="${tuning}/bert_log_v1_i${i}_t${j}.txt"
    echo ${fname} ${i} ${j}
    python ./multi_instance_BERT.py --model ./out/test_bert_b1.so --num-instances ${i} --num-threads ${j} > ${fname}
  done
done
