# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import onnx
import os
import platform
import time
import argparse
import sys

file_path = os.path.realpath(__file__)
demo_folder = os.path.dirname(file_path)

if not 'TVM_HOME' in os.environ:
  print("'TVM_HOME' is not set so the script path is used as reference to the TVM project.")
  tvm_path = os.path.join(demo_folder, "..", "..")
  os.environ['TVM_HOME']=tvm_path
else:
  tvm_path = os.environ['TVM_HOME']
sys.path.append(os.path.join(tvm_path, 'python'))


import tvm
from tvm.relay.analysis import get_total_mac_number
from tvm import relay, auto_scheduler
from tvm.relay import transform

def getInput(model, batch_size):
    retval = {}
    for input in model.graph.input:
        nm = input.name
        shape = []
        # # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")): # known dimension
                    shape.append(int(d.dim_value))
                elif (d.HasField("dim_param")): # unknown dimension with symbolic name
                    # workaround for now!
                    shape.append(batch_size)
        retval[nm] = shape
    return retval


# target = "llvm -mcpu=skylake-avx512"
target = "llvm -mcpu=znver3"
# target_host = "llvm -mcpu=znver3"

parser = argparse.ArgumentParser()
parser.add_argument("--onnx-model", help="reference to the onnx model", default="__data/dlrm_onnx/dlrm_s_pytorch_0505.onnx")
parser.add_argument("--tuning-log-file", required=False, help="path to the tuning log", default=None)
parser.add_argument("--output-name", required=False, help="name of compiled model", )
parser.add_argument("--batch-size", type=int, required=False, help="optional, batch size for the model", default=1)
args = parser.parse_args()


def load_onnx(model_path, batch_size=128):

    onnx_model = onnx.load(model_path)
    shape_dict = getInput(onnx_model, batch_size)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    seq = tvm.transform.Sequential(
      [
          # transform.InferType(),
          # transform.FoldConstant(),
          # transform.SimplifyInference(),
          # transform.FoldScaleAxis(),
          # transform.DynamicToStatic(),
          # transform.AlterOpLayout(),
          # transform.PartitionGraph(),
          transform.InferType(),
          transform.DynamicToStatic()
      ]
    )

    mod = seq(mod)

    macs = get_total_mac_number(mod["main"])
    print(f"total MACs for batch {batch_size} : {macs}")

    return mod, params


def compile(mod, params, output_name):
    so_ext = "dylib" if platform.system() == "Darwin" else "so"
    export_lib_path = f"__prebuilt/{output_name}.{so_ext}"
    export_json_path = f"__prebuilt/{output_name}.json"
    export_param_path = f"__prebuilt/{output_name}.npz"

    start_timestamp = time.time()
    if args.tuning_log_file is not None:
        with auto_scheduler.ApplyHistoryBest(args.tuning_log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                json, lib, param = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3, config={}):    
            json, lib, param = relay.build(mod, target=target, params=params)
    
    compile_dur = time.time() - start_timestamp
    print(f"Compilation time : {compile_dur}")

    lib.export_library(export_lib_path)

    with open(export_json_path, "w") as fp:
        fp.write(json)

    start_timestamp = time.time()
    param_map = {key:val.asnumpy() for key, val in param.items()}
    np.savez(export_param_path, **param_map)

    param_dump_dur = time.time() - start_timestamp
    print(f"Param dump time : {param_dump_dur}")


def main():
    os.makedirs("__prebuilt", exist_ok=True)

    mod, params = load_onnx(model_path=args.onnx_model, batch_size=args.batch_size)
    compile(mod, params, output_name=f"{args.output_name}_b{args.batch_size}")


if __name__ == "__main__":
    main()
