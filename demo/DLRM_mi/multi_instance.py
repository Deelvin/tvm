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

from audioop import mul
import numpy as np
import os
import time
import warnings
import argparse
import threading
import random
from queue import Queue
from tvm.runtime.vm import VirtualMachine

import tvm
from tvm.contrib import graph_executor, dyn_batch_slicer

from .models import get_cpu_info, get_so_ext, models, default_model_path

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
    tasks = Queue(maxsize=num_instance * 1)
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

    real_duration_sec = time.time() - start_time

    # grab performance results
    total_count = 0
    total_duration_us = 0.0
    for count, duration in zip(res_count, res_time):
        total_count += count
        total_duration_us += duration

    avg_latency = total_duration_us / total_count * 1000
    avg_throughput = total_count / real_duration_sec

    return avg_latency, avg_throughput


def get_macs(model_path):
    model_macs_file = model_path[:-len(get_so_ext()) - 1] + ".macs.txt"

    with open(model_macs_file, "r") as f:
        macs = int(f.read())

    return macs

_set_affinity = tvm._ffi.get_global_func("tvm.set_affinity", allow_missing=True)
_setup_thr_arena_mutex = threading.Lock()

def setup_thread_scheme(arena_ids):
    assert isinstance(arena_ids, list)

    _setup_thr_arena_mutex.acquire()
    os.environ["TVM_BIND_THREADS"] = "0"
    os.environ["TVM_NUM_THREADS"] = str(len(arena_ids))
    os.environ["OMP_NUM_THREADS"] = str(len(arena_ids))

    if _set_affinity:
        _set_affinity(arena_ids)
    else:
        warnings.warn(
            "Looks like your version of TVM doesn't have 'tvm.set_affinity' function. "
            "Thread pinning has a positive effect of final performance. "
            "Please apply patch from __patches/set_affinity.patch"
        )

    _setup_thr_arena_mutex.release()

class AffinityScheme:
    def __init__(self, scheme):
        self.scheme_ = scheme

    def __len__(self):
        return len(self.scheme_)

    def __getitem__(self, idx):
        return self.scheme_[idx]

    def __str__(self):
        arena_sizes = [len(indxs) for indxs in self.scheme_]
        min_arena_size = min(arena_sizes)
        max_arena_size = max(arena_sizes)
        return f"{len(self.scheme_)}-{min_arena_size}/{max_arena_size}"

    @staticmethod
    def unisize_scheme(num_inst, num_thr):
        arena_sizes = [num_thr] * num_inst
        return AffinityScheme._scheme(arena_sizes)

    @staticmethod
    def balanced_scheme(num_inst, total_num_thr=os.cpu_count()):
        arena_sizes = [total_num_thr // num_inst] * num_inst

        for i in range(total_num_thr % num_inst):
            arena_sizes[i] += 1

        return AffinityScheme._scheme(arena_sizes)

    @staticmethod
    def _scheme(arena_sizes):
        # setup for google cloud
        scl = 1 # 2 if get_cpu_info().hyper_threading else 1
        scheme = []
        arena_start = 0
        for arena_size in arena_sizes:
            scheme.append([i * scl for i in range(arena_start, arena_start + arena_size)])
            arena_start += arena_size

        return AffinityScheme(scheme)


def get_batch_size(g_mod):
    in0 = g_mod.get_input(0).numpy()
    return in0.shape[0]


main_g_mod = None
shared_weight_names = None

def bench_round(affinity_scheme, args):
    model_path = args.model_path
    model_json_file = model_path[:-len(get_so_ext())] + "json"
    use_vm = False
    if os.path.isfile(model_json_file):
        model_param_file = model_path[:-len(get_so_ext())] + "npz"
        model_param_file_common = model_path[:-len(get_so_ext()) - 1] + "_common.npz"
    else:
        use_vm = True
        head, _ = os.path.split(model_path)
        model_param_file = os.path.join(head, "vm_exec_code.ro")
        model_param_file_common = os.path.join(head, "consts")

    num_inst = len(affinity_scheme)
    lib = tvm.runtime.load_module(model_path)


    json = None
    code = None
    weights_are_ready = threading.Barrier(num_inst, timeout=6000)  # 10 min like infinity value
    if not use_vm:
        with open(model_json_file, "r") as json_f:
            json = json_f.read()
    else:
        code = bytearray(open(model_param_file, "rb").read())


    def init_f(idx):
        setup_thread_scheme(affinity_scheme[idx])

        global main_g_mod
        global shared_weight_names
        dyn_batch_config = None
        input_gen = None

        if args.model_name in models.keys():
            _, _, input_gen, dyn_batch_config = models[args.model_name]
        if not use_vm:
            g_mod = graph_executor.create(json, lib, tvm.cpu())
            if idx == 0:
                if main_g_mod is None:
                    start_timestamp = time.time()

                    print("Loading params...")
                    if not use_vm:
                        params = np.load(model_param_file)
                        g_mod.set_input(**params)
                    params_names = {}
                    for param_name in params:
                        params_names[param_name] = tvm.nd.empty([1])  # stub tensors

                    if os.path.exists(model_param_file_common):
                        print("detect common npz")
                        com_params = np.load(model_param_file_common)
                        filter = ["p0", "p2", "p4", "p33", "p35", "p37", "p39", "p41"]
                        for key in com_params:
                            if key in filter:
                                continue
                            g_mod.set_input(key, com_params[key])
                            params_names[key] = tvm.nd.empty([1])

                    load_param_dur = time.time() - start_timestamp
                    print(f"Load param time : {load_param_dur}")

                    main_g_mod = g_mod
                    shared_weight_names = tvm.runtime.save_param_dict(params_names)
                else:
                    g_mod.share_params(main_g_mod, shared_weight_names)
        else:
            mod = tvm.runtime.vm.Executable.load_exec(code, lib)
            mod.load_late_bound_consts(model_param_file_common)
            g_mod = VirtualMachine(mod, tvm.cpu())
            if input_gen:
                inpt = input_gen(args.batch_size)
                g_mod.invoke("main", **inpt)
            else:
                if 'inputs' in dir(args):
                    g_mod.invoke("main", **args.inputs)
                else:
                    print("ERROR: input is not defined.")
                    exit(0)
        # share weights from instance id==0
        weights_are_ready.wait()
        if not use_vm and idx != 0:
            # timeDelay = random.uniform(0, 5)
            # time.sleep(timeDelay)
            g_mod.share_params(main_g_mod, shared_weight_names)


            # If original batch is not equal to requested will use dyn_batch_slicer
            if args.batch_size != get_batch_size(g_mod):
                g_mod = dyn_batch_slicer.create(g_mod, config=dyn_batch_config)

            for key, data in input_gen(args.batch_size).items():
                g_mod.set_input(key, data)
        for i in range(3):
            g_mod.run()
        return g_mod

    def process_f(g_mod):
        g_mod.run()

    avg_latency, avg_throughput = runer_queued(init_f, process_f, duration_sec=args.trial_time, num_instance=num_inst)

    print(f"CFG:{affinity_scheme}, AVG_LAT:{avg_latency:.2f}, AVG_THR:{avg_throughput:.2f}", flush = True)


def main_call(args):
    print((args))
    if args.model_path == "default":
        if args.model_name in default_model_path.keys():
            args.model_path = default_model_path[args.model_name]
        else:
            print("ERROR : only dlrm, bert and resnet models can have defult path")

    # get_macs(args.model_path)

    num_cpu = os.cpu_count()
    unisize = AffinityScheme.unisize_scheme
    balanced = AffinityScheme.balanced_scheme

    # 1-INST
    # for num in range(1, num_cpu + 1):
    #     bench_round(unisize(1, num))

    # 1-THR
    # for num in range(1, num_cpu + 1):
    #     bench_round(unisize(num, 1))

    # 2-THR
    # for num in range(1, num_cpu // 2 + 1):
    #     bench_round(unisize(num, 2))

    # FULL
    # for num in range(1, num_cpu // 2 + 1):
    #     bench_round(balanced(num, num_cpu))
    for num in range(1, num_cpu  + 1):
        for j in range(1, num_cpu // num + 1):
            bench_round(unisize(num, j), args)

    # bench_round(unisize(args.num_instances, args.num_threads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="model name [resnet50, ]", default="dlrm")
    parser.add_argument("--model-path", help="path to compiled model", default="__prebuilt/dlrm_avx512.so")
    parser.add_argument("--num-instances", type=int, help="path to compiled model", default=1)
    parser.add_argument("--num-threads", type=int, help="path to compiled model", default=1)
    parser.add_argument("--batch-size", type=int, help="Batch to process with", default=1)
    parser.add_argument("--trial-time", type=int, help="Time of one trial (sec)", default=10)
    args = parser.parse_args()
    main_call(args)
