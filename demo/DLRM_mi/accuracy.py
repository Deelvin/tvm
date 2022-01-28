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
from tvm.contrib import graph_executor

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to compiled model", default="__prebuilt/dlrm_avx512.so")
parser.add_argument("--ref-data", required=False, help="path to collection of reference input/output", default="__data/mlperf_data.npz")
args = parser.parse_args()

def load_graph(model_path):
    model_json_file = model_path[:-2] + "json"
    model_param_file = model_path[:-2] + "npz"
    lib = tvm.runtime.load_module(model_path)
   
    with open(model_json_file, "r") as json_f: 
        json = json_f.read()
   
    g_module = graph_executor.create(json, lib ,tvm.cpu())
    
    start_timestamp = time.time()
    
    params = np.load(model_param_file)
    g_module.set_input(**params)

    load_param_dur = time.time() - start_timestamp
    print(f"Load param time : {load_param_dur}")

    return g_module


def get_batch_size(g_mod):
    in0 = g_mod.get_input(0).numpy()
    return in0.shape[0]


def load_data(dataset_path):
    with np.load(dataset_path) as data:
        X_cat = data["X_cat"]
        X_int = data["X_int"]
        ref = data["torch_res"]
        y = data["target"]
        return X_int, X_cat, ref, y


def main():
    X_int, X_cat, ref, target = load_data(args.ref_data)
    g_mod = load_graph(args.model)

    dataset_size = ref.shape[0]
    batch = get_batch_size(g_mod)
    
    print(f"Processing {dataset_size} elements with batch {batch} ...") 

    for iter in range(dataset_size // batch):
        # make proper shapes
        idx = iter * batch
        t = target[idx:idx + batch]
        ref_out = ref[idx:idx + batch]
        x_in = X_int[idx:idx + batch, :]
        ls_i_in = np.transpose(X_cat[idx:idx + batch, :])
        ls_o_in = np.array([range(batch) for _ in range(26)])
        
        # make proper types
        x_in = x_in.astype("float32")
        ls_i_in = ls_i_in.astype("int64")
        ls_o_in = ls_o_in.astype("int64")

        g_mod.set_input("input.1", x_in)
        g_mod.set_input("lS_o", ls_o_in)
        g_mod.set_input("lS_i", ls_i_in)
        g_mod.run()
        
        out = g_mod.get_output(0).numpy()
        out = np.reshape(out, (-1,))
        
        assert np.allclose(out, ref_out)

    print("Looks like everything is good") 


if __name__ == "__main__":
    main()
