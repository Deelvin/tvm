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

from models import get_host_isa, get_host_target, get_so_ext, models, default_model_path
from tvm import relay, auto_scheduler
from tvm.relay.analysis import get_total_mac_number
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True, help="Model name [resnet, dlrm, bert]")
parser.add_argument("--model-path", required=False, help="path to the original model", default="default")
parser.add_argument("--tuning-log-file", required=False, help="path to the tuning log", default="default")
parser.add_argument("--batch-size", required=False, type=int, help="optional, batch size for the model", default=1)
parser.add_argument("--with-dnnl", required=False, help="name of compiled model", default=0)
args = parser.parse_args()


def compile_mod(mod, params, output_name, opt_level):
    target = get_host_target()
    so_ext = get_so_ext()

    os.makedirs(f"__prebuilt/{output_name}", exist_ok=True)

    export_lib_path = f"__prebuilt/{output_name}/{output_name}.{so_ext}"
    export_json_path = f"__prebuilt/{output_name}/{output_name}.json"
    export_param_path = f"__prebuilt/{output_name}/{output_name}.npz"
    export_macs_path = f"__prebuilt/{output_name}/{output_name}.macs.txt"

    macs = get_total_mac_number(mod["main"])

    if args.with_dnnl:
        mod = partition_for_dnnl(mod)
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv2d_transpose": ["NHWC", "default"],
        "nn.upsampling": ["NHWC", "default"],
        "vision.roi_align": ["NHWC", "default"],
    }

    if args.tuning_log_file is not None and os.path.exists(args.tuning_log_file):
        with auto_scheduler.ApplyHistoryBest(args.tuning_log_file):
            with tvm.transform.PassContext(opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True, "relay.FuseOps.max_depth": 30}):
              seq = tvm.transform.Sequential(
                    [
                        relay.transform.InferType(),
                        relay.transform.ConvertLayout(desired_layouts),
                        relay.transform.EliminateCommonSubexpr(),
                        relay.transform.FoldConstant(),
                    ]
                )
              irmod = seq(mod)
              f_lib = relay.build(irmod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=opt_level, config={}):
            f_lib = relay.build(mod, target=target, params=params)

    g_json = f_lib.get_graph_json()
    g_lib = f_lib.get_lib()
    g_params = f_lib.get_params()

    g_lib.export_library(export_lib_path)

    with open(export_json_path, "w") as fp:
        fp.write(g_json)

    with open(export_macs_path, "w") as fp:
        fp.write(str(macs))

    # filter = ["p0", "p2", "p4", "p33", "p35", "p37", "p39", "p41"]
    # param_map = {key: val.asnumpy() for key, val in g_params.items() if key in filter}
    param_map = {key: val.asnumpy() for key, val in g_params.items()}
    np.savez(export_param_path, **param_map)


def main():
    isa = get_host_isa()
    target = get_host_target()

    if args.tuning_log_file == "default":
        args.tuning_log_file = f"__data/common.{isa}.tune"

    if args.model_path == "default":
        args.model_path = default_model_path[args.model_name]

    print()
    print("=================================================")
    print(f" Model Name : {args.model_name}")
    print(f" Model Path : {args.model_path}")
    print(f" Batch Size : {args.batch_size}")
    print(f" Precision  : {'INT8' if args.model_name.endswith('i8') else 'FP32'}")
    print("=================================================")
    print(f" Compiler : {target}")
    print(f" ISA      : {isa}")
    print(f" Tuning   : {args.tuning_log_file}")
    print("=================================================")
    print()

    loader, opt_level, _, _ = models[args.model_name]
    mod, params = loader(args.model_path, args.batch_size)

    export_name = f"{args.model_name}_b{args.batch_size}_{isa}"
    compile_mod(mod, params, output_name=export_name, opt_level=opt_level)


if __name__ == "__main__":
    main()
