Important note: This branch contains some modifications within graph_executor.cc which will be removed in future.

# Octomized model inference

    To work with Octomized models 
run_octomized_model.py [-h] [--token TOKEN] [--model-uuid MODEL_UUID] --system-name SYSTEM_NAME [--download-log DOWNLOAD_LOG] --model-name MODEL_NAME [--model-path MODEL_PATH]
                              [--batch-size BATCH_SIZE] [--trial-time TRIAL_TIME]
# DEMO setup (optional)

The folder tvm/demo contains a set of scripts which are used for DLRM and BERT models inference performance analysis.
Environment setup script:
```
python prepare.py --model [DLRM, BERT, all]
```

This script does following things:
* Uploads MLCommons Inference repository
* Uploads data for BERT model and creates docer container for the BERT model inference.
* Uploads DLRM repository if DRRM model is selected for the setup.
  * Uploads 100 GB DLRM model. The model will be stored in tvm/demo/DLRM/model folder.
  * Performs weights extraction for further inference. The converted weights are stored in tvm/demo/DLRM/converted folder.
* Updates MLCommons Inference repository to run DLRM and/or BERT models using tvm inference.
* Generates fake dataset for DLRM inference.

Note: it is not required to setup envirenment variables according to /inference/recommendation/dlrm/pytorch/README.md because
these variables are set at the beginning of run_local.sh script.


