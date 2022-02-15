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
import os
import time
import argparse
import tvm

from models import models, get_host_target, get_so_ext
from tvm import relay, auto_scheduler
from tvm.relay.analysis import get_total_mac_number
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", help="reference to the original model", default="__data/dlrm_onnx/dlrm_s_pytorch_0505.onnx")
parser.add_argument("--model-name", help="Model name [resnet, dlrm, bert]", default="dlrm")
parser.add_argument("--tuning-log-file", required=False, help="path to the tuning log", default=None)
parser.add_argument("--output-name", required=False, help="name of compiled model", )
parser.add_argument("--batch-size", type=int, help="optional, batch size for the model", default=100)
parser.add_argument("--with-dnnl", required=False, help="name of compiled model", )
args = parser.parse_args()


def compile_mod(mod, params, output_name, opt_level):
    target = get_host_target()
    so_ext = get_so_ext()

    os.makedirs(f"__prebuilt/{output_name}", exist_ok=True)

    export_lib_path = f"__prebuilt/{output_name}/{output_name}.{so_ext}"
    export_json_path = f"__prebuilt/{output_name}/{output_name}.json"
    export_param_path = f"__prebuilt/{output_name}/{output_name}.npz"

    if args.with_dnnl:
        mod = partition_for_dnnl(mod)

    start_timestamp = time.time()
    if args.tuning_log_file is not None:
        with auto_scheduler.ApplyHistoryBest(args.tuning_log_file):
            with tvm.transform.PassContext(opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}):
                json, lib, param = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=opt_level, config={}):
            json, lib, param = relay.build(mod, target=target, params=params)

    compile_dur = time.time() - start_timestamp
    print(f"Compilation time : {compile_dur}")

    lib.export_library(export_lib_path)

    with open(export_json_path, "w") as fp:
        fp.write(json)

    start_timestamp = time.time()
    param_map = {key: val.asnumpy() for key, val in param.items()}
    np.savez(export_param_path, **param_map)

    param_dump_dur = time.time() - start_timestamp
    print(f"Param dump time : {param_dump_dur}")


def main():
    loader, _, _ = models[args.model_name]
    mod, params = loader(args.model_path, args.batch_size)

    macs = get_total_mac_number(mod["main"])

    print()
    print("===================================")
    print(f" Model Name : {args.model_name}")
    print(f" Model Path : {args.model_path}")
    print(f" Batch Size : {args.batch_size}")
    print(f" Precision  : {'INT8' if args.model_name.endswith('i8') else 'FP32'}")
    print(f" MACs       : {macs}")
    print("===================================")
    print(f" Tuning stat: {args.tuning_log_file}")
    print("===================================")
    print()
    opt_level = 3
    if args.model_name.find('resnet') != -1:
        opt_level = 2
    export_name = f"{args.model_name}_b{args.batch_size}"
    compile_mod(mod, params, output_name=export_name, opt_level)


if __name__ == "__main__":
    main()
