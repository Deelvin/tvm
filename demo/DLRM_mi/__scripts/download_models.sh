echo Model donwloading script

if ! command -v wget &> /dev/null
then
    echo "[ERROR] wget could not be found. Please install it."
    exit 1
fi

function download {
  local name_=$1
  local url_=$2

  mkdir __models/${name_} -p
  wget ${url_} -q --show-progress -P __models/${name_}
#   echo "${name_} ${url_}"
}

function extract_tar {
  local tar_name_=$1
  tar -xvzf ${tar_name_}
}

echo "ResNet50_fp32 ..."
url="https://zenodo.org/record/4735647/files/resnet50_v1.onnx"
name="resnet50_fp32"
download ${name} ${url}

echo "ResNet50_int8 ..."
url="https://zenodo.org/record/4589637/files/resnet50_INT8bit_quantized.pt"
name="resnet50_int8"
download ${name} ${url}

echo "BERT_fp32 ..."
url="https://zenodo.org/record/3733910/files/model.onnx"
name="bert_fp32"
download ${name} ${url}

echo "BERT_int8 ..."
url="https://zenodo.org/record/3750364/files/bert_large_v1_1_fake_quant.onnx"
name="bert_int8"
download ${name} ${url}


# echo "DLRM_fp32 ..."
# url="https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.onnx.tar"
# name="dlrm" 




# echo ResNet50_int8
# echo Bert_fp32