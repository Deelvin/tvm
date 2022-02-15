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
import warnings
import argparse
import threading
from queue import Queue

import tvm
from tvm.contrib import graph_executor

from models import get_so_ext, models

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", help="model name [resnet50, ]", default="dlrm")
parser.add_argument("--model-path", help="path to compiled model", default="__prebuilt/dlrm_avx512.so")
parser.add_argument("--num-instances", type=int, help="path to compiled model", default=1)
parser.add_argument("--num-threads", type=int, help="path to compiled model", default=1)
parser.add_argument("--hyper-threading", type=bool, help="path to compiled model", default=True)
args = parser.parse_args()


def runer_loop(init_f, process_f, tasks_queue, idx, initialized_barier, res_count, res_time):
    ctx = init_f(idx)

    initialized_barier.wait()

    tot_count = 0
    start_timestamp = time.time()

    if not isinstance(ctx, tuple):
        ctx = (ctx,)

    while True:
        item = tasks_queue.get()
        if item is None:
            # None in the queue indicates the parent want us to exit
            tasks_queue.task_done()
            break
        process_f(*ctx)
        tot_count += 1
        tasks_queue.task_done()

    tot_duration = time.time() - start_timestamp

    res_count[idx] = tot_count
    res_time[idx] = tot_duration


def runer_queued(init_f, process_f, *, duration_sec=5, num_instance=8):
    tasks = Queue(maxsize=num_instance * 15)
    workers = []

    initialized_barier = threading.Barrier(num_instance + 1, timeout=6000)  # 100 min like infinity value

    res_count = [0] * num_instance
    res_time = [0.0] * num_instance

    # start worker loops in sub threads
    for idx in range(num_instance):
        worker = threading.Thread(target=runer_loop,
                                  args=(init_f, process_f, tasks, idx, initialized_barier, res_count, res_time))
        worker.daemon = True
        workers.append(worker)
        worker.start()

    initialized_barier.wait()

    # submit tasks in queue while reach duration time
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        tasks.put(1)

    # send termination message
    for _ in range(num_instance):
        tasks.put(None)

    for worker in workers:
        worker.join()

    # grab performance results
    total_count = 0
    total_duration = 0.0
    for count, duration in zip(res_count, res_time):
        total_count += count
        total_duration += duration

    latency = total_duration / total_count * 1000
    throughput = int(total_count / duration_sec)
    print(f"NUM_INST :{num_instance}, AVG_LATENCY:{latency:.2f} ms, AVG_THR:{throughput}")

    return latency, throughput


def get_batch_size(g_mod):
    in0 = g_mod.get_input(0).numpy()
    return in0.shape[0]


def main():
    model_path = args.model_path
    model_json_file = model_path[:-len(get_so_ext())] + "json"
    model_param_file = model_path[:-len(get_so_ext())] + "npz"

    lib = tvm.runtime.load_module(model_path)

    weights_are_ready = threading.Barrier(args.num_instances, timeout=6000)  # 10 min like infinity value

    with open(model_json_file, "r") as json_f:
        json = json_f.read()

    # thread engine preparations
    os.environ["TVM_BIND_THREADS"] = "0"
    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    set_affinity = tvm._ffi.get_global_func("tvm.set_affinity", allow_missing=True)
    if set_affinity is None:
        warnings.warn(
            "Looks like your version of TVM doesn't have 'tvm.set_affinity' function. "
            "Thread pinning has a positive effect of final performance. "
            "Please apply patch from __patches/set_affinity.patch"
        )

    def init_f(idx):
        if set_affinity is not None:
            set_affinity(int(idx * args.num_threads), args.num_threads, args.hyper_threading)

        global main_g_mod
        global shared_weight_names
        g_mod = graph_executor.create(json, lib, tvm.cpu())

        if idx == 0:
            start_timestamp = time.time()

            print("Loading params...")
            params = np.load(model_param_file)
            g_mod.set_input(**params)

            params_names = {}
            for param_name in params:
                params_names[param_name] = tvm.nd.empty([1])  # stub tensors

            load_param_dur = time.time() - start_timestamp
            print(f"Load param time : {load_param_dur}")

            main_g_mod = g_mod
            shared_weight_names = tvm.runtime.save_param_dict(params_names)

        weights_are_ready.wait()

        # share weights from instance id==0
        if idx != 0:
            g_mod.share_params(main_g_mod, shared_weight_names)

        batch = get_batch_size(g_mod)
        _, input_gen = models[args.model_name]

        for key, data in input_gen(batch).items():
            g_mod.set_input(key, data)

        return g_mod

    def process_f(g_mod):
        g_mod.run()

    runer_queued(init_f, process_f, duration_sec=30, num_instance=args.num_instances)


if __name__ == "__main__":
    main()
