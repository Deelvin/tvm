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
import subprocess
import tvm
from tvm.relay.analysis import get_total_mac_number
from tvm import relay, auto_scheduler
from tvm.relay import transform

def getCPUVendor():
    cpu_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    spl = cpu_info.split('\n')
    print(len(spl))
    for i in range(len(spl)):
        if spl[i].find('Model name') != -1:
            print(spl[i])
            if spl[i].find('AMD') != -1:
                target = "llvm -mcpu=znver3"
                target_host = "llvm -mcpu=znver3"
            else:
                target = "llvm -mcpu=skylake-avx512"
                target_host = "llvm -mcpu=skylake-avx512"
            return target, target_host

target, _ = getCPUVendor()

parser = argparse.ArgumentParser()
parser.add_argument("--onnx-model", help="reference to the onnx DLRM model", default="__data/dlrm_onnx/dlrm_s_pytorch_0505.onnx")
parser.add_argument("--tuning-log-file", required=False, help="path to the tuning log", default=None)
parser.add_argument("--output-name", required=False, help="name of compiled model", )
parser.add_argument("--batch-size", type=int, help="optional, batch size for the model", default=100)
args = parser.parse_args()


def load_onnx(model_path, batch_size=128):
    shape_dict = {
        "input.1": (batch_size, 13),
        "lS_o": (26, batch_size),
        "lS_i": (26, batch_size)
    }

    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    
    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)

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
