DLRM with Multi-Instance execution
==================================

# Prerequisite
You should have:
* Downloaded ONNX DLRM model. https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.onnx.tar 
* Built TVM (-DUSE_LLVM=ON, -DUSE_GRAPH_EXECUTOR=ON, -DUSE_OPENMP=gnu) with applied patches form "__patches"
* All TVM dependencies to perform compilation and tuning


# Step 1. Compilation
```bash
python compile.py --onnx-model ${ONNX_DLRM_MODEL_PATH} --batch-size=100 \ 
                  --tuning-log-file=__data/dlrm_tune_100.tune_log --output-name=dlrm_avx512  
```

By default, it will use tuning statistic from pre collected file __data/dlrm_tune_100.log. Skip it to use default 
code generation. Batch size is 100 by default, may be changed. resulting files will be located in folder "__prebuilt".


# Step 2. Check accuracy
```bash
python accuracy.py --model __prebuilt/dlrm_avx512_b100.so --ref-data __data/mlperf_data_small.npz                     
```
There is a pre collected reference inputs/outputs from MLPerf inference scripts. This scripts use original pytorch 
inference on real dataset. File mlperf_data_small.npz contains first 32768 items form day_23 which is usually used like
a test dataset.


# Step 3. Check performance in Multi-Instance mode
```bash
python multi_instance.py --model __prebuilt/dlrm_avx512_b100.so --num-instances 26 --num-threads 2                     
```

This script will check performance of several simultaneously working tvm graph executors. You may configure number of 
internal TVM threads and number of instances.    

This script uses a tvm.set_affinity extension to manually bind worker threads to cores. So it's highly recommended 
to apply patch "__patches/set_affinity.patch".


# Results and conclusions
For testing we used machine Azure Ev5 (104 vCPU).
```text
    Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
    Num of sockets: 2
    Hyper-threading: ON
    Base freq: 2.8 GHz
    Boost freq: 3.5 GHz
    RAM: 672 GiB
```

Multi-Instance demo script with batch=100 num_threads=2 num_instances=26 shows throughput ~9250. Which can be translated 
to 3425 QPS for offline mode of MLperf measure script. This value is ~55% of peak performance of Azure Ev5_104 system.

TVM with OpenMP has a slightly better performance results. So we recommend to use OMP threading engine. 