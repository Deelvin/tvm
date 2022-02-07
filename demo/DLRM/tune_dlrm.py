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

"""
This  script perfroms tuning of DLRM model using local tuner for Zen 3 HW.
"""

import os
import sys
import argparse

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
import onnx
from tvm import relay, auto_scheduler
from tvm.relay import transform
from tvm.contrib import graph_executor
from params_demo import *

def checkDir(pth):
  if not os.path.exists(pth):
    os.makedirs(pth)

def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")

    tuner  = auto_scheduler.TaskScheduler(tasks, task_weights)
    builder = auto_scheduler.LocalBuilder(build_func="default", n_parallel=1, timeout=10000)
    tune_option = auto_scheduler.TuningOptions(
        builder = builder,
        num_measure_trials=ITERATIONS,
        num_measures_per_round = 8,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )
    tuner.tune(tune_option)

def doPreprocess(mod):
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
            transform.DynamicToStatic(),
            transform.AlterOpLayout(),
            transform.PartitionGraph(),
        ]
    )
    mod = seq(mod)
    return mod

parser = argparse.ArgumentParser()
parser.add_argument("--onnx-model", help="optional, reference to the onnx DLRM model", default='')
parser.add_argument("--output-log", required=True, help="path to the output log file")
parser.add_argument("--output-folder", required=True, help="path to the output library and json files")
parser.add_argument("--batch-size", help="optional, batch size for the model", default=128)

args = parser.parse_args()
target, target_host = getCPUVendor()

log_file = args.output_log
out_dir = args.output_folder
onnx_model = args.onnx_model
if onnx_model == '':
    file_path = os.path.realpath(__file__)
    demo_folder = os.path.dirname(file_path)
    model_dir = os.path.join(tvm_path, 'demo', MODEL_SUFF)
    onnx_model = os.path.join(model_dir, ONNX_FILE_NAME)
    if os.path.isfile(onnx_model) != True:
        print("ERROR:  there is no onnx file  {} in this folder {}".format(ONNX_FILE_NAME, model_dir))
        quit()
if args.batch_size:
    BATCH_SIZE = int(args.batch_size)
    if BATCH_SIZE > 0:
        shape_dict["input.1"] = (BATCH_SIZE, 13)
        shape_dict["lS_o"] = (26, BATCH_SIZE)
        shape_dict["lS_i"] = (26, BATCH_SIZE)

onnx_model = onnx.load(onnx_model)

ctx = tvm.cpu(0)
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
mod = doPreprocess(mod)

del onnx_model

tasks, task_weights = auto_scheduler.extract_tasks(
    mod["main"], params, target=target, target_host=target_host, include_simple_tasks = False)

run_tuning(tasks, task_weights, log_file)

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True
                                                        }):
        json, lib, param = relay.build(mod, target=target, params=params)
    model_path = os.path.join(out_dir, "saved_model_{}_b{}.tar".format(ITERATIONS, BATCH_SIZE))
    lib.export_library(model_path)
tvm.runtime.load_module(model_path)
jsonName = os.path.join(out_dir, "model_serialized_tuned_{}_b{}.json".format(ITERATIONS, BATCH_SIZE))
with open(jsonName, "w") as fp:
    fp.write(json)
print("output library file: ", model_path)
print("output json file: ", jsonName)
