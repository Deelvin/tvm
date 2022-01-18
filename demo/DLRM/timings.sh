#!/bin/bash
AR=(1 2 4 8 16 24 32 48 64 96 120)
length=${#AR[@]}
for (( i = 0; i < length; i++ )); do
  export TVM_NUM_THREADS=${AR[$i]}
  export OMP_NUM_THREADS=${AR[$i]}
  python ./run_dlrm.py $1 $2
done