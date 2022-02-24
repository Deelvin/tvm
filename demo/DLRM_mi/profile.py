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
import time
import argparse

import tvm
from tvm.contrib.debugger import debug_executor
from models import models

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True, help="model name")
parser.add_argument("--model-path", required=False, help="model name", default="default")
args = parser.parse_args()


def load_graph(model_path):
    model_json_file = model_path[:-2] + "json"
    model_param_file = model_path[:-2] + "npz"
    lib = tvm.runtime.load_module(model_path)
   
    with open(model_json_file, "r") as json_f: 
        json = json_f.read()
   
    g_module = debug_executor.create(json, lib ,tvm.cpu())
    
    start_timestamp = time.time()
    
    params = np.load(model_param_file)
    g_module.set_input(**params)

    load_param_dur = time.time() - start_timestamp
    print(f"Load param time : {load_param_dur}")

    _, _, input_gen, _ = models[args.model_name]

    batch = get_batch_size(g_module)

    for key, data in input_gen(batch).items():
        g_module.set_input(key, data)

    return g_module


def get_batch_size(g_mod):
    in0 = g_mod.get_input(0).numpy()
    return in0.shape[0]


def main():
    g_mod = load_graph(args.model_path)

    set_affinity = tvm._ffi.get_global_func("tvm.set_affinity", allow_missing=True)
    if set_affinity:
        set_affinity(0, 60, False)
        print("Affinity !!!")
    # g_mod.run()

    score = g_mod.run_individual(500, repeat=1, min_repeat_ms=0)
    # print(score)


if __name__ == "__main__":
    main()
