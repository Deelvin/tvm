#!/bin/bash
tuning="tuning"
mkdir -p "./${tuning}"
pth_twm="${PWD}/../../"
length_thr=$(nproc)
length_thr=$((length_thr + 1))
echo $length_thr
export TVM_HOME=$pth_twm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
i=1
# export OMP_NUM_THREADS=8
model_pth="/home/ubuntu/dev/tvm/demo"
# python ./tune_resnet50.py --onnx-model ${model_pth}/resnet50_v1.onnx --output-log ./resnet50.json --output-folder ./out
# python ./compile.py --model-path ${model_pth}/resnet50_v1.onnx --model-name resnet50 --tuning-log-file ./resnet50.json --output-name resnet50 --batch-size 1
for (( i = 1; i < length_thr; i++ )); do
  threads=$(($(nproc)/i))
  threads=$((threads + 1))
  for (( j = 1; j < threads; j++ )); do
    fname="${tuning}/log_RESNET50_v1_i${i}_t${j}.txt"
    echo ${fname} ${i} ${j}
    python ./multi_instance.py --model-path ./__prebuilt/resnet_b1_avx2/resnet50_b1_avx2.so --model-name resnet --num-instances ${i} --num-threads ${j} > ${fname}
  done
done
upper=$(nproc)
upper=$((upper + 1))
for (( i = length_thr; i < upper; i++ )); do
  for (( j = 1; j < 3; j++ )); do
    fname="${tuning}/log_RESNET50_v1_i${i}_t${j}.txt"
    echo ${fname} ${i} ${j}
    python ./multi_instance.py --model-path ./__prebuilt/resnet_b1_avx2/resnet50_b1_avx2.so --model-name resnet --num-instances ${i} --num-threads ${j} > ${fname}
  done
done
