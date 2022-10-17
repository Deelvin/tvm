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
import pytest

import tvm
from tvm import tir
from tvm.script import tir as T

from tvm import autotvm


from tvm import te

@autotvm.template("test_sram")
def test_sram(data_shape, weight_shape, output_shape, dtype):
    data = te.placeholder(data_shape, name="data", dtype=dtype)
    weight = te.placeholder(weight_shape, name="weight", dtype=dtype)

    res = te.compute(output_shape, lambda i: data[i] * weight[i], name="res")
    cfg = autotvm.get_config()
    (d,) = res.op.axis
    cfg.define_split("tile_d", d, num_outputs=4)

    s = te.create_schedule(res.op)
    return s, [data, weight, res]


def test_sram_autotvm():
    target = tvm.target.Target("llvm", host="llvm")
    data_shape = (1024,)
    weight_shape = (1024,)
    output_shape = (1024,)
    dtype = "float32"

    with target:
        s, (data, weight, res) = test_sram(data_shape, weight_shape, output_shape, dtype)
        mod_lowered = tvm.lower(s, [data, weight, res], simple_mode=True)
        print("LOWER", mod_lowered)
        mod = tvm.build(s, [data, weight, res], target=target)

    task = autotvm.task.create("test_sram", args=(data_shape, weight_shape, output_shape, dtype), target=target)
    print("CONFIG",task.config_space)
    tasks = [task]

    model_name = "test_sram"
    log_file = "%s.log" % model_name
    tuning_option = {"log_filename": log_file,"tuner": "xgb","early_stopping": None,"measure_option": autotvm.measure_option(builder=autotvm.LocalBuilder(),runner=autotvm.LocalRunner(number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True),),}
    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    def run_tuning(tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"):
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            # create tuner
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)
            # do tuning
            n_trial = len(task.config_space)
            tuner_obj.tune(n_trial=n_trial,early_stopping=early_stopping,measure_option=measure_option,callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix),autotvm.callback.log_to_file(log_filename),],)

    # run_tuning(tasks, **tuning_option)
    
    calc_size = tvm.tir.analysis.calculate_inout_tensors_bytes(mod_lowered["main"])
    import numpy as np
    ref_size = sum(list(map(np.prod, [data_shape, weight_shape, output_shape]))) * 4
    print("cacl: ", calc_size, "ref: ", ref_size)
    assert calc_size == ref_size

    # VerifySRAMLimit

    # print("ref", data_shape[0] * weight_shape[0] * output_shape[0] * 32)

        # def gpu_verify_pass(**kwargs):
    #     """Verify the validity of a gpu kernel.
    #     This pass will check memory usage and number of threads per block.
    #     """

    #     def verify_pass(f, *_):
    #         count = tvm.tir.analysis.calculate_inout_tensors_bytes(f, kwargs)
    #         # target

    #         if count < 10000:
    #             raise InstantiationError("Skipped because of invalid gpu kernel")
    #         return f

    #     return tvm.tir.transform.prim_func_pass(verify_pass, opt_level=0)




    # def get_verify_pass(valid, **kwargs):
    #     def _fverify(f, *_):
    #         # valid[0] = tvm.tir.analysis.calculate_inout_tensors_bytes(f, kwargs)
    #         valid[0] = tvm.tir.analysis.calculate_inout_tensors_bytes(f)
    #         if valid[0] == 12288:
    #         # if valid[0] != 12288
    #             from tvm.autotvm.task.space import InstantiationError
    #             raise InstantiationError("Skipped because of invalid sram size")
            
    #         return f

    #     return tvm.tir.transform.prim_func_pass(_fverify, opt_level=0)


    # valid = [None]
    # with tvm.transform.PassContext(
    #     config={
    #         "tir.add_lower_pass": [
    #             (
    #                 2,
    #                 get_verify_pass(
    #                     valid
    #                 ),
    #             )
    #         ]
    #     }
    # ):
    #     m = tvm.build(s, [data, weight, res], target)
    #     print(m)
    # print("valid", valid)

def test_sram_autoscheduler():
    from tvm import te
        # @tvm.auto_scheduler.register_workload
    # def test_sram():
    #     data = te.placeholder(data_shape, name="data", dtype=dtype)
    #     weight = te.placeholder(weight_shape, name="weight", dtype=dtype)
    #     res = te.compute(output_shape, lambda i: data[i] * weight[i], name="res")
    #     return[data, weight, res]
    # target = tvm.target.Target("llvm")
    # task = tvm.auto_scheduler.SearchTask(func=test_sram, args=(), target=target)



    @tvm.auto_scheduler.register_workload  # Note the auto_scheduler decorator
    def matmul_add(N, L, M, dtype):
        A = te.placeholder((N, L), name="A", dtype=dtype)
        B = te.placeholder((L, M), name="B", dtype=dtype)
        C = te.placeholder((N, M), name="C", dtype=dtype)

        k = te.reduce_axis((0, L), name="k")
        matmul = te.compute(
            (N, M),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name="matmul",
            attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
        )
        out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

        return [A, B, C, out]
    target = tvm.target.Target("llvm")
    # N = L = M = 4 # no hangs
    N = L = M = 1024 # hangs with limit eq 1024*10
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)
    
    args_ = matmul_add(N, L, M, "float32")
    sout = te.create_schedule(args_[3].op)

    print("PRELowered TIR:")
    print(tvm.lower(sout, args_, simple_mode=True))


    print("Computational DAG:")
    print(task.compute_dag)
    log_file = "matmul.json"
    tune_option = tvm.auto_scheduler.TuningOptions(
        num_measure_trials=151,
        builder=tvm.auto_scheduler.LocalBuilder(n_parallel=1),
        measure_callbacks=[tvm.auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

if __name__ == "__main__":
    # test_sram_autotvm()
    test_sram_autoscheduler()






   





