
# Model tuning and inference automation
  The script run_octomized_model.py applied to work with onnx models to perform following steps:
  1. Upload model to Octo.ai for tuning.
  2. Download results and generate performance metrics for diffetent scenarios.
  3. Generates latency/throughput points cloud to evlaute testing system behaviour.

# Possible testiing scenarious:
## Generate results for the particular onnx model.
  The commandlien example

```python ./run_octomized_model.py  --platform=gcp_milan_n2d-highmem-48 --model-path ./ssd_mobilenet_v1_coco_2018_01_28.onnx --model-name mobilenet --batch-size  1 --inputs-descriptor=./ssd_inputs.json```

where:
  platform: available platform for tuning. This is mandatory parameter.
  model-path: path to the onnx model.
  model-name: model name for internal usage. This name will be printed as title for output metrics graph. Note: 5 names are reserved for particular models from MLPerf test suite: bert, bert_i8, dlrm, resnet and resnet_i8 (resnet50)
  batch-size: definition of desired batch size.
  inputs-descriptor: optional parametes which defines onnx model inputs.

  The example of json file:

```
{
  "input:0" : {
    "input_dtype" : "float32",
    "shape" : [1, 3, 300, 300]
  },
  "indices:0" : {
    "input_dtype" : "int64",
    "shape" : [100]
  }
}
```

The script perfroms step 1 and provides following information at the end of execution:


```Please wait notification from Octo.ai and run following commandline:```
```python ./run_octomized_model.py --platform=gcp_milan_n2d-highmem-48 --workflow-uuid=<generated uuid> --model-name=mobilenet --batch-size=1 --inputs-descriptor=./ssd_inputs.json```

So after receiving e-mail notification it is necessary to run provided generated commandline and wait for the results.
Currently 2 files are generated:
output_Log.json - file with experiment results (the name of file can be changed if user adds --output-log commandline)
res_<platfortm>_<model_name>.png