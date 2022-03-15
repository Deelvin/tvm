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
import psutil

from models import models, default_model_path, get_host_isa, get_host_target
from tvm import auto_scheduler, relay

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True, help="Model name [resnet, dlrm, bert]",)
parser.add_argument("--model-path", required=False, help="reference to the original model", default="default")
parser.add_argument("--batch-size", required=False, type=int, help="optional, batch size for the model", default=1)
args = parser.parse_args()


def tune_mod(mod, params, output_name, opt_level):
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv2d_transpose": ["NHWC", "default"],
        "nn.upsampling": ["NHWC", "default"],
        "vision.roi_align": ["NHWC", "default"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.ConvertLayout(desired_layouts),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.FoldConstant(),
        ]
    )
    mod = seq(mod)

    os.makedirs(f"__tuning/{output_name}", exist_ok=True)
    log_file = f"__tuning/{output_name}/{output_name}.log"

    target = get_host_target()

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target=target, target_host=target, opt_level=opt_level)
    # tasks = tasks[5:6]
    # task_weights = task_weights[5:6]

    # # XXX
    # for i, t in enumerate(tasks):
    #     print(f"=== task #{i} {t.workload_key} ===")
    #     print(t.compute_dag)

    # exit()
    # # XXX

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    builder = auto_scheduler.LocalBuilder(build_func="default", timeout=30)
    # runner = auto_scheduler.LocalRunner(repeat=10, min_repeat_ms=300, timeout=30, enable_cpu_cache_flush=True)
    runner = auto_scheduler.LocalRunner(number=5, min_repeat_ms=200, timeout=30, enable_cpu_cache_flush=False)

    tune_option = auto_scheduler.TuningOptions(
        builder=builder,
        runner=runner,
        num_measure_trials=25000,
        num_measures_per_round=64,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )

    tuner.tune(tune_option)


def main():
    if args.model_path == "default":
        args.model_path = default_model_path[args.model_name]

    isa = get_host_isa()

    loader, opt_level, _, _ = models[args.model_name]
    mod, params = loader(args.model_path, args.batch_size)

    print()
    print("===================================")
    print(f" Model Name : {args.model_name}")
    print(f" Model Path : {args.model_path}")
    print(f" Batch Size : {args.batch_size}")
    print(f" Precision  : {'INT8' if args.model_name.endswith('i8') else 'FP32'}")
    print("===================================")
    print()

    export_name = f"{args.model_name}_b{args.batch_size}_{isa}"
    tune_mod(mod, params, output_name=export_name, opt_level=opt_level)


if __name__ == "__main__":
    main()
