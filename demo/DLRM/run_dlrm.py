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
This  script perfroms DLRM model execution.
"""

import os
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

import argparse
import numpy as np
import os
import tvm
from tvm import relay, auto_scheduler
import onnx
from tvm.relay import transform
from tvm.contrib import graph_executor
import time
from tvm.contrib.debugger import debug_executor
from tvm.relay.op.contrib.register import get_pattern_table
from params_demo import *

parser = argparse.ArgumentParser()

parser.add_argument("--onnx-file", required=True, help="path to onnx model file")
parser.add_argument("--log-file", required=True, help="path to the log file with tuning information.")
parser.add_argument("--test-data", help="optional, path to the test data.")
parser.add_argument("--output-folder", help="optional, path to the output library and json files")
parser.add_argument("--batch-size", help="optional, batch size for the model", default=128)

args = parser.parse_args()
if 'TVM_NUM_THREADS' in os.environ:
    print("current threads = {}".format(os.environ['TVM_NUM_THREADS']))

name = args.onnx_file
log_file = args.log_file
output_folder = args.output_folder
test_data = args.test_data
onnx_model = onnx.load(name)

if args.batch_size:
    BATCH_SIZE = int(args.batch_size)
    if BATCH_SIZE > 0:
        shape_dict["input.1"] = (BATCH_SIZE, 13)
        shape_dict["lS_o"] = (26, BATCH_SIZE)
        shape_dict["lS_i"] = (26, BATCH_SIZE)

ctx = tvm.cpu(0)
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

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
dnnl_codegen = tvm.support.libinfo().get("USE_DNNL_CODEGEN", "OFF")
if dnnl_codegen == "ON":
    patternTBL = get_pattern_table("dnnl")
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
            transform.DynamicToStatic(),
            transform.AlterOpLayout(),
            transform.FoldConstant(),
            transform.MergeComposite(patternTBL),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph()
        ]
    )
mod = seq(mod)

use_tuned = True
use_debug = False
del onnx_model
if use_tuned:
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
            json, lib, param = relay.build(mod, target=target, params=params)

else:
    with tvm.transform.PassContext(opt_level=3, config={}):
        json, lib, param = relay.build(mod, target=target, params=params)

print("model built.")
if dnnl_codegen == "OFF": # may not be compiled
    if (output_folder != None and os.path.exists(output_folder)):
        model_path = os.path.join(output_folder, "saved_model_{}_b{}.tar".format(ITERATIONS, BATCH_SIZE))
        lib.export_library(model_path)
        tvm.runtime.load_module(model_path)
        jsonName = os.path.join(output_folder, "model_serialized_tuned_{}_b{}.json".format(ITERATIONS, BATCH_SIZE))
        with open(jsonName, "w") as fp:
            fp.write(json)
        print("so.and json created.")

if not use_debug:
    model = graph_executor.create( json, lib, ctx)
else:
    model = debug_executor.create(json, lib, ctx)
print("executor created.")
model.set_input(**param)

np.random.seed(0)

inpt = (np.zeros(shape = shape_dict["input.1"])).astype(dtype_dict["input.1"])
ls_i = (np.zeros(shape = shape_dict["lS_i"])).astype(dtype_dict["lS_i"])
ls_o = (np.zeros(shape = shape_dict["lS_o"])).astype(dtype_dict["lS_o"])

for i in range(shape_dict["lS_o"][0]):
    for j in range(BATCH_SIZE):
        ls_o[i, j] = j

model.set_input("input.1", tvm.nd.array(inpt))
model.set_input("lS_i", tvm.nd.array(ls_i))
model.set_input("lS_o", tvm.nd.array(ls_o))

print("load test data")
test_folder = os.path.join(demo_folder, TEST_DATA_SUFF)
if test_data != None and os.path.exists(test_data):
    test_folder = test_data

batch_input = np.fromfile(os.path.join(test_folder, "batch_dense_X_ref"), dtype=np.float32)
batch_lsi = np.fromfile(os.path.join(test_folder, "batch_lS_i_ref"), dtype=np.int64)
batch_referense_res = np.fromfile(os.path.join(test_folder, "batch_res_ref"), dtype=np.float32)

input_batch = batch_input.shape[0] // shape_dict["input.1"][1]
lsi_batch = batch_lsi.shape[0] // shape_dict["lS_i"][0]

do_check = True
if input_batch != lsi_batch or batch_referense_res.shape[0] != input_batch:
    print("ERROR: incorrect test data.")
    do_check = False
    count = 1000
else:
    new_input = batch_input.reshape((input_batch, shape_dict["input.1"][1]))
    new_lsi = batch_lsi.reshape((shape_dict["lS_i"][0], lsi_batch))
    count = input_batch // BATCH_SIZE

# warmup
if not use_debug:
    model.run()
total_time = 0
max_error = 0
avg_error = 0
LOOPS_COUNT = 20
runs = 0
full_start = time.perf_counter()
if not use_debug:
    for k in range(LOOPS_COUNT):
        total_time = 0
        max_error = 0
        avg_error = 0
        for i in range(count):
            if do_check:
                off_1 = i * BATCH_SIZE
                off_2 = (i + 1) * BATCH_SIZE
                inpt = new_input[off_1:off_2, :]
                ls_i = new_lsi[:, off_1:off_2]
                model.set_input("input.1", tvm.nd.array(inpt))
                model.set_input("lS_i", tvm.nd.array(ls_i))

            start = time.perf_counter()
            model.run()
            runs += 1
            end = time.perf_counter()
            total_time += (end - start)
            if do_check:
                res = model.get_output(0).asnumpy()
                for j in range(BATCH_SIZE):
                    val = abs(res[j] - batch_referense_res[i *  BATCH_SIZE + j])
                    avg_error += val
                    max_error = max(max_error, val)
else:
    for i in range(count):
        start = time.perf_counter()
        res = model.profile()
        end = time.perf_counter()
        print(res)
full_end = time.perf_counter()
if do_check:
    print("avg error: {}".format(avg_error / (count * BATCH_SIZE)))
    print("max error: {}".format(max_error))
print("BATCH_SIZE = ", BATCH_SIZE, ", count = ", count, ", runs = ", runs)
# print("    average inference time = {} sec.".format((total_time)/runs))
# print("    average inference per input = {} sec.".format((total_time)/runs/BATCH_SIZE))
# print("    total execution time = {} sec.".format(full_end - full_start))
# print("    total avg execution time = {} sec.".format(full_end - full_start)/runs)
