Important note: This branch contains some modifications within graph_executor.cc which will be removed in future.

# DEMO setup

The floder DLRM contains a set of scripts which are used for DLRM model inference performance analysis.
Environment setup script:
```
python prepare.py
```

This script does following things:
* Uploads DLRM repository and MLCommons Inference repository.
* Uploads 100 GB DLRM model. The model will be stored in tvm/demo/DLRM/model folder.
* Performs weights extraction for further inference. The converted weights are stored in tvm/demo/DLRM/converted folder

# DRM model tuning

```
python tune_dlrm.py --output-log OUTPUT_LOG --output-folder OUTPUT_FOLDER [--onnx-model ONNX_MODEL]
```

Where:
OUTPUT_LOG - is a name of log file with tuning information.
OUTPUT_FOLDER - folder where generated library and json file will be stored  after the script finished.
ONNX_MODEL - optional parameter which should be used if model is not allocated in tvm/demo/DLRM/model folder.

After script execution the OUTPUT_FOLDER will contain following files:
* saved_model_XXX.tar
* saved_model_XXX.tar.so
* model_serialized_tuned_XXX.json
Where XXX is a number of iterations for tuning. The default value for this parameter is 3000 but it can be reconfigured in params_demo.py file.

# Native Inference pipeline

Compile rules:

```
mkdir build
cd ./build
cmake ..
```
Command line:
'''
dlrm_infer <reference to model so> <reference to model json> <reference to weights folder> <reference to test data folder>
'''
Where:
<reference to model so> is a result  of DLRM model tuning (see #DLRM model tuning). By default it is OUTPUT_FOLDER/saved_model_XXX.tar.so
<reference to model json> is a result  of DLRM model tuning (see #DLRM model tuning). By default it is OUTPUT_FOLDER/model_serialized_tuned_XXX.json
<reference to weights folder> Model weights folder. By default it should be: tvm/demo/DLRM/converted
<reference to test data folder> Folder with testing data. By default it should be: tvm/demo/DLRM/test_data
