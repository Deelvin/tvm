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

# ICE

# @T.prim_func
# def primfunc_global_allocates(placeholder_144: T.handle, placeholder_145: T.handle, placeholder_146: T.handle, T_cast_48: T.handle) -> None:
#     # function attr dict
#     T.func_attr({"global_symbol": "fused_nn_conv2d_add_cast_fixed_point_multiply_clip_cast_cast_13", "tir.noalias": True})
#     placeholder_147 = T.match_buffer(placeholder_144, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
#     placeholder_148 = T.match_buffer(placeholder_145, [4608], dtype="int16", elem_offset=0, align=64, offset_factor=1)
#     placeholder_149 = T.match_buffer(placeholder_146, [512], dtype="int32", elem_offset=0, align=64, offset_factor=1)
#     T_cast_49 = T.match_buffer(T_cast_48, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
#     # body
#     PaddedInput_22 = T.decl_buffer([131072], "int16")
#     DepthwiseConv2d_9 = T.decl_buffer([100352], "int32")
#     for i1_29, i2_39, i3_40 in T.grid(16, 16, 512):
#         PaddedInput_22[(((i1_29*8192) + (i2_39*512)) + i3_40)] = T.if_then_else(((((1 <= i1_29) and (i1_29 < 15)) and (1 <= i2_39)) and (i2_39 < 15)), placeholder_147[((((i1_29*7168) + (i2_39*512)) + i3_40) - 7680)], T.int16(0), dtype="int16")
#     for i_9, j_9, c_9 in T.grid(14, 14, 512):
#         DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = 0
#         for di_9, dj_9 in T.grid(3, 3):
#             DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = (DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] + (PaddedInput_22[(((((i_9*8192) + (di_9*8192)) + (j_9*512)) + (dj_9*512)) + c_9)].astype("int32")*placeholder_148[(((di_9*1536) + (dj_9*512)) + c_9)].astype("int32")))
#     for ax1_27, ax2_28, ax3_30 in T.grid(14, 14, 512):
#         DepthwiseConv2d_9[(((ax1_27*7168) + (ax2_28*512)) + ax3_30)] = (DepthwiseConv2d_9[(((ax1_27*7168) + (ax2_28*512)) + ax3_30)] + placeholder_149[ax3_30])
#     for i1_30, i2_40, i3_41 in T.grid(14, 14, 512):
#         DepthwiseConv2d_9[(((i1_30*7168) + (i2_40*512)) + i3_41)] = T.q_multiply_shift(DepthwiseConv2d_9[(((i1_30*7168) + (i2_40*512)) + i3_41)], 1269068532, 31, -4, dtype="int32")
#     for i1_31, i2_41, i3_42 in T.grid(14, 14, 512):
#         DepthwiseConv2d_9[(((i1_31*7168) + (i2_41*512)) + i3_42)] = T.max(T.max(DepthwiseConv2d_9[(((i1_31*7168) + (i2_41*512)) + i3_42)], 255), 0)
#     for ax1_28, ax2_29, ax3_31 in T.grid(14, 14, 512):
#         PaddedInput_22[(((ax1_28*7168) + (ax2_29*512)) + ax3_31)] = DepthwiseConv2d_9[(((ax1_28*7168) + (ax2_29*512)) + ax3_31)].astype("uint8")
#     for ax1_29, ax2_30, ax3_32 in T.grid(14, 14, 512):
#         T_cast_49[(((ax1_29*7168) + (ax2_30*512)) + ax3_32)] = PaddedInput_22[(((ax1_29*7168) + (ax2_30*512)) + ax3_32)].astype("int16")



# @pytest.mark.parametrize("alignment,size,consts", [(1, 663552, 0), (10, 663560, 0)])
# def test_global_allocates(alignment, size, consts):
#     primfunc = primfunc_global_allocates
#     assert tvm.tir.analysis.calculate_constant_bytes(primfunc, alignment) == consts
#     assert tvm.tir.analysis.calculate_workspace_bytes(primfunc, alignment) == size



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
    # N = L = M = 4
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
        # builder=tvm.auto_scheduler.LocalBuilder("default", n_parallel=1),
        # runner=tvm.auto_scheduler.LocalRunner(timeout=10), n_parallel=1
        builder=tvm.auto_scheduler.LocalBuilder(n_parallel=1),
        # runner=tvm.auto_scheduler.LocalRunner(n_parallel=1),
        measure_callbacks=[tvm.auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

# @tvm.testing.requires_hexagon
# def test_hexa(hexagon_session: Session)):
from tvm import topi
import numpy as np
import tvm.contrib.hexagon
def test_hexa():
    print("test_hexa")
    dtype = "uint8"
    in_shape = (1, 56, 56, 32)

    data_in = np.ones(in_shape).astype(dtype)

    A = te.placeholder(shape=in_shape, name="A", dtype=dtype)

    C = topi.nn.pad(A, [0, 1, 1, 0], [0, 1, 1, 0], pad_value=0)

    target_hexagon = "llvm"
    # target_hexagon = tvm.target.hexagon("v68")
    with tvm.target.Target(target_hexagon):
        fschedule = topi.hexagon.schedule_pad
        s = fschedule(C)

    func = tvm.build(s, [A, C], tvm.target.Target(target_hexagon, host=target_hexagon), name="pad")
    # print(func.get_source())
    print(func.get_function("pad"))
    print(dir(func))
    print(repr(func))
    print(dir(func["pad"]))
    print(repr(func["pad"]))
    print(func["pad"].handle)


def test_texture_scope():
    @tvm.script.ir_module
    class PlusOneMultTwo:
        @T.prim_func
        def main(a: T.handle, b: T.handle) -> None:
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # A = T.match_buffer(a, (128, 128))
            # B = T.match_buffer(b, (128, 128))
            # for i, j in T.grid(128, 128):
            #     with T.block("B"):
            #         vi, vj = T.axis.remap("SS", [i, j])
            #         B[vi, vj] = A[vi, vj] * 2.0

            A = T.match_buffer(a, (128, 128, 4), dtype="float32", scope="global.texture")
            B = T.alloc_buffer((128, 128, 4), dtype="float32", scope="global.texture")
            C = T.match_buffer(b, (128, 128, 4), dtype="float32", scope="global")
            D = T.alloc_buffer((128, 128, 4), dtype="float32", scope="global")
            for block_idx in T.thread_binding(0, 128, thread="blockIdx.x"):
                for thread_idx in T.thread_binding(0, 128, thread="threadIdx.x"):
                    for k in T.serial(4):
                        with T.block("B"):
                            vb, vt, vk = T.axis.remap("SSS", [block_idx, thread_idx, k])
                            B[vb, vt, vk] = A[vb, vt, vk] + T.float32(1)
            for block_idx in T.thread_binding(0, 128, thread="blockIdx.x"):
                for thread_idx in T.thread_binding(0, 128, thread="threadIdx.x"):
                    for k in T.serial(4):
                        with T.block("C"):
                            vb, vt, vk = T.axis.remap("SSS", [block_idx, thread_idx, k])
                            C[vb, vt, vk] = B[vb, vt, vk] * T.float32(2)
            for block_idx in T.thread_binding(0, 128, thread="blockIdx.x"):
                for thread_idx in T.thread_binding(0, 128, thread="threadIdx.x"):
                    for k in T.serial(4):
                        with T.block("D"):
                            vb, vt, vk = T.axis.remap("SSS", [block_idx, thread_idx, k])
                            D[vb, vt, vk] = C[vb, vt, vk] * T.float32(2)

    sch = tir.Schedule(PlusOneMultTwo, debug_mask="all")

    def schedule_block(block):
        _, _, inner = sch.get_loops(block)
        sch.vectorize(inner)

    schedule_block(sch.get_block("B"))
    schedule_block(sch.get_block("C"))

    target = tvm.target.Target("opencl")
    mod = tvm.build(sch.mod["main"], target=target)
    mod_lowered = tvm.lower(sch.mod["main"], [], simple_mode=True)
    print("LOWER", mod_lowered)

    # print(tvm.lower(sout, args_, simple_mode=True))
    # print(mod.entry_func)
    # print(dir(mod.entry_func))
    calc_size = tvm.tir.analysis.calculate_inout_tensors_bytes(mod_lowered["main"])
    calc_alloc_size = tvm.tir.analysis.calculate_allocated_bytes(mod_lowered["main"])
    # calc_alloc_size = tvm.tir.calculate_allocated_bytes(mod_lowered["main"])
    print("calc_size", calc_size)
    print("calc_size", calc_alloc_size)
    print("expt_size", ((128 *  128 * 4) * 8)* 3)

if __name__ == "__main__":


    # import tvm.relay as relay
    # import tvm

    # # model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
    # import onnx
    # model_path = "C:\\Users\\icemist\\Downloads\\srgan_obfuscated.onnx"
    # onnx_model = onnx.load(model_path)
    # # shape_dict = (1, 128,128, 3)
    # input_name = "input"
    # shape_dict = {input_name: (1, 128,128, 3)}
    # mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # from tvm import relay, auto_scheduler
    # target = tvm.target.Target("llvm")
    # print("extract_tasks")
    # tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    # print("end extract_tasks")
    # # print(list(mod.functions.values())[0])
    # print("tasks", len(tasks))
    # print(dir(tasks[0]))
    # for id, task in enumerate(tasks):
    #     print(id, '\n', task.compute_dag)
    #     print()
    # # print(mod["main"])
    # exit()
    test_texture_scope()
    # test_hexa()
    # test_sram_autotvm()
    # test_sram_autoscheduler()






   





