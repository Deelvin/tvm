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
import argparse
import tvm

from models import models, get_host_target
from tvm import relay, auto_scheduler
from tvm.relay.analysis import get_total_mac_number
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", help="reference to the original model", default="__data/dlrm_onnx/dlrm_s_pytorch_0505.onnx")
parser.add_argument("--model-name", help="Model name [resnet, dlrm, bert]", default="dlrm")
parser.add_argument("--tuning-log-file", required=False, help="path to the tuning log", default=None)
parser.add_argument("--batch-size", type=int, help="optional, batch size for the model", default=100)
args = parser.parse_args()


def tune_mod(mod, params, output_name):
    os.makedirs(f"__tuning/{output_name}", exist_ok=True)
    log_file = f"__tuning/{output_name}/{output_name}.log"

    target = get_host_target()

    # TODO:  opt_level=2. Because of limitation. opt_level=3 lead to change dense->matmul with wrong shapes.
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target=target, target_host=target, opt_level=2)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    builder = auto_scheduler.LocalBuilder(build_func="default", n_parallel=1, timeout=10)
    runner = auto_scheduler.LocalRunner(min_repeat_ms=500, timeout=10)

    tune_option = auto_scheduler.TuningOptions(
        builder=builder,
        runner=runner,
        num_measure_trials=3000,
        num_measures_per_round=4,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )

    tuner.tune(tune_option)


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
    print()

    export_name = f"{args.model_name}_b{args.batch_size}"
    tune_mod(mod, params, output_name=export_name)


if __name__ == "__main__":
    main()
