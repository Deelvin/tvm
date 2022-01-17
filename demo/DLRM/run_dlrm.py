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
  print(tvm_path)
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
from params_demo import *

parser = argparse.ArgumentParser()

parser.add_argument("--onnx-file", required=True, help="path to onnx model file")
parser.add_argument("--log-file", required=True, help="path to the log file with tuning information.")
parser.add_argument("--output-folder", required=True, help="path to the output library and json files")

args = parser.parse_args()

name = args.onnx_file
log_file = args.log_file
output_folder = args.output_folder
onnx_model = onnx.load(name)

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
mod = seq(mod)
use_tuned = True
use_debug = False
index = 1000
if use_tuned:
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
            json, lib, param = relay.build(mod, target=target, params=params)
else:
    with tvm.transform.PassContext(opt_level=3, config={}):
        json, lib, param = relay.build(mod, target=target, params=params)
del onnx_model
if not use_debug:
    model = graph_executor.create( json, lib, ctx)
else:
    model = debug_executor.create(json, lib, ctx)
model.set_input(**param)

if output_folder != '' and os.path.exists(output_folder):
    model_path = os.path.join(output_folder, "saved_model_{}.tar".format(ITERATIONS))
    lib.export_library(model_path)
    tvm.runtime.load_module(model_path)
    jsonName = os.path.join(output_folder, "model_serialized_tuned_{}.json".format(ITERATIONS)); 
    with open(jsonName, "w") as fp:
        fp.write(json)

np.random.seed(0)

for iname, ishape in shape_dict.items():
    np_data = (np.random.uniform(size=ishape)).astype(dtype_dict[iname])
    model.set_input(iname, tvm.nd.array(np_data))

count = 1000
# warmup
if not use_debug:
    model.run()
start = time.perf_counter()
if not use_debug:
    for i in range(count):
        model.run()
else:
    for i in range(count):
        res = model.profile()
        print(res)

end = time.perf_counter()
print("time = {}".format((end - start)/count))
