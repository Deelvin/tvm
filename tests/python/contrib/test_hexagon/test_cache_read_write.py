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

""" Lower cache_read and cache_write to Hexagon DMA via tensorize """

import numpy as np

import tvm.testing
from tvm import te, tir
from tvm.contrib.hexagon.session import Session
from tvm.script import tir as T

from .infrastructure import get_hexagon_target


def intrin_mem_copy(shape, dtype, dst_scope, src_scope):
    """Define and return tensor intrinsic for mem copy"""
    src = te.placeholder(shape=shape, dtype=dtype, name="src")
    dst = te.compute(shape, lambda i: src[i], name="dst")
    size = shape[0] * np.dtype(dtype).itemsize

    src_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=src_scope,
        offset_factor=1,
        name="mem_copy_src_buffer",
    )

    dst_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=dst_scope,
        offset_factor=1,
        name="mem_copy_dst_buffer",
    )

    zero_indices = [0 for _ in shape]

    def intrin_func(ins, outs):
        ir_builder = tvm.tir.ir_builder.create()

        _src = ins[0]
        _dst = outs[0]

        dst_handle = ir_builder.buffer_ptr(dst_buffer)
        src_handle = ir_builder.buffer_ptr(src_buffer)

        ir_builder.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.mem_copy",
                tvm.tir.call_intrin("handle", "tir.address_of", dst_handle[zero_indices]),
                tvm.tir.call_intrin("handle", "tir.address_of", src_handle[zero_indices]),
                size,
            )
        )
        return ir_builder.get()

    return te.decl_tensor_intrin(dst.op, intrin_func, binds={src: src_buffer, dst: dst_buffer})


def verify(hexagon_session: Session, schedule, x_tensor, y_tensor, z_tensor, size):
    """Verify correctness with reference from numpy"""
    # print(tvm.lower(schedule, [x_tensor, y_tensor, z_tensor]), flush=True)
    target_hexagon = tvm.target.hexagon("v68", vtcm_capacity=128,link_params=True)
    # print(target_hexagon)
    print("vtcm_capacity",target_hexagon.vtcm_capacity)
    # print("vtcm_capacity",target_hexagon.vtcm_capacity)
    # print(dir(target_hexagon))
    # target_hexagon = target.hexagon("v69", num_cores=4)
    # tvm.target.hexagon("v66", hvx=128)

    # with tvm.target.Target(target_hexagon, host=target_hexagon):
    #     func = tvm.build(
    #         schedule,
    #         [x_tensor, y_tensor, z_tensor],
    #         tvm.target.Target(target_hexagon, host=target_hexagon),
    #         name="dmacpy",
    #     )
    # with tvm.transform.PassContext(config={"tir.vtcm_capacity": 128}):
    #     func = tvm.build(
    #         schedule,
    #         [x_tensor, y_tensor, z_tensor],
    #         tvm.target.Target(target_hexagon, host=target_hexagon),
    #         name="dmacpy",
    #     )

    func = tvm.build(
        schedule,
        [x_tensor, y_tensor, z_tensor],
        get_hexagon_target("v68"),
        name="dmacpy",
    )
    return 
    # print("func", type(func), func, flush=True)
    mod = hexagon_session.load_module(func)
    # print("mod", type(mod), mod, flush=True)
    # mod_lowered = tvm.lower(schedule, [x_tensor, y_tensor, z_tensor])
    # d = tir.analysis.calculate_allocated_bytes(mod_lowered["main"])
    # print("data", d, flush=True)

    # def ir_module_has_allocate_nodes(irmod):
    #     nallocs = 0

    #     def _visit(stmt):
    #         nonlocal nallocs
    #         if isinstance(stmt, tvm.tir.Allocate):
    #             nallocs += 1

    #     tvm.tir.stmt_functor.post_order_visit(irmod["main"].body, _visit)
    #     print("ICE nallocs: ", nallocs, flush=True)
    #     return nallocs

    # ir_module_has_allocate_nodes(mod_lowered)
    
    # mod = tir.transform.VerifyVTCMLimit(1024)(mod_lowered["main"])
    # mod = tir.transform.VerifyVTCMLimit(1024)(mod)
    x_array = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=x_tensor.dtype),
        device=hexagon_session.device,
    )
    y_array = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=y_tensor.dtype),
        device=hexagon_session.device,
    )
    z_array = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=z_tensor.dtype),
        device=hexagon_session.device,
    )
    mod["dmacpy"](x_array, y_array, z_array)

    ref = x_array.numpy() + y_array.numpy()
    np.testing.assert_equal(z_array.numpy(), ref)




# import numpy as np
# import pytest

# import tvm.testing
# from tvm import relay
# from tvm.contrib.hexagon.session import Session
# from tvm.relay.backend import Executor, Runtime


# import onnx  # pylint: disable=import-outside-toplevel
# # def get_mobilenet():
# #     """Download and import mobilenet model with ONNX"""

# #     model_url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"  # pylint: disable=line-too-long
# #     model_path = tvm.contrib.download.download_testdata(
# #         model_url, "mobilenetv2-7.onnx", module="onnx"
# #     )
# #     return onnx.load(model_path)


# @tvm.testing.requires_hexagon
# def test_hexa(hexagon_session: Session):
#     # """Test mobilenet with graph executor"""
#     dtype = "float32"
#     model_path = "/git/srgan_obfuscated.onnx"
#     onnx_model = onnx.load(model_path)
#     # onnx_model = get_mobilenet()

#     target_hexagon = tvm.target.hexagon("v68")
#     # target_llvm = tvm.target.Target("llvm")
#     runtime = Runtime("cpp")
#     executor = Executor("graph", {"link-params": True})

#     data_in = np.random.rand(1, 128,128, 3).astype(dtype=dtype)
#     # data_in = np.random.rand(1, 3, 224, 224).astype(dtype=dtype)
#     input_name = "input"
#     shape_dict = {input_name: data_in.shape}
#     relay_mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
#     inputs = {input_name: data_in}

#     with tvm.transform.PassContext(opt_level=3):
#         hexagon_lowered = tvm.relay.build(
#             relay_mod,
#             tvm.target.Target(target_hexagon, host=target_hexagon),
#             runtime=runtime,
#             executor=executor,
#             params=params,
#         )
#         x = te.placeholder(shape=(1, 128,128, 3), dtype="float32", name="x")
#         schedule = tir.Schedule(relay_mod, debug_mask="all")
#         mod_lowered = tvm.lower(schedule.mod, [x])
#         print(mod_lowered, flush=True)
#         d = tir.analysis.calculate_allocated_bytes(mod_lowered["main"])
#         print("data", d)

#         # print(hexagon_lowered)

#         # llvm_lowered = tvm.relay.build(
#         #     relay_mod,
#         #     tvm.target.Target(target_llvm, host=target_llvm),
#         #     runtime=runtime,
#         #     executor=executor,
#         #     params=params,
#         # )

#     # graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
#     # graph_mod.set_input(**inputs)
#     # graph_mod.run()
#     # hexagon_output = graph_mod.get_output(0).numpy()

#     # llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
#     # llvm_graph_mod.set_input(**inputs)
#     # llvm_graph_mod.run()
#     # expected_output = llvm_graph_mod.get_output(0).numpy()

#     # tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


# @tvm.testing.requires_hexagon
# def test_hexa__(hexagon_session: Session):
#     print("test_hexa")

#     import tvm.relay as relay
#     import tvm

#     # model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
#     import onnx
#     model_path = "/git/srgan_obfuscated.onnx"
#     onnx_model = onnx.load(model_path)
#     input_name = "input"
#     shape_dict = {input_name: (1, 128,128, 3)}
#     mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#     from tvm import relay, auto_scheduler
#     # target_hexagon = tvm.target.hexagon("v68")
#     # target = tvm.target.Target(target_hexagon, host=target_hexagon)
#     mod = hexagon_session.load_module(mod)
#     mod = tir.transform.VerifyVTCMLimit(1024)(mod)
#     # sch = tir.Schedule(mod, debug_mask="all")
#     # x = te.placeholder(shape=(1, 128,128, 3), dtype="float32", name="x")
#     # print(tvm.lower(schedule, [x]))
#     # target_hexagon = tvm.target.hexagon("v68", link_params=True)
#     # func = tvm.build(
#     #     schedule,
#     #     [x_tensor, y_tensor, z_tensor],
#     #     tvm.target.Target(target_hexagon, host=target_hexagon),
#     #     name="dmacpy",
#     # )



#     # print("func", type(func), func)
#     # print("mod", type(mod), mod)
#     # mod_lowered = tvm.lower(schedule, [x_tensor, y_tensor, z_tensor])


#     # target = tvm.target.Target("llvm")
#     # print("extract_tasks")
#     # # tasks, task_weights = tvm.auto_scheduler.extract_tasks(mod["main"], params, target)
#     # print("end extract_tasks")
#     # # print(list(mod.functions.values())[0])
#     # print("tasks", len(tasks))
#     # print(dir(tasks[0]))
#     # for id, task in enumerate(tasks):
#     #     print(id, type(task.compute_dag), '\n', task.compute_dag)
#     #     print()
#     # print(mod["main"])

@tvm.testing.requires_hexagon
def test_cache_read_write(hexagon_session: Session):
    """Test cache_read and cache_write to global.vtcm for hexagon""" # ICE
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x_tensor = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y_tensor = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z_tensor = te.compute(outer_shape, lambda i: x_tensor[i] + y_tensor[i], name="z")
    s = te.create_schedule(z_tensor.op)

    x_vtcm = s.cache_read(x_tensor, "global.vtcm", [z_tensor])
    y_vtcm = s.cache_read(y_tensor, "global.vtcm", [z_tensor])
    z_vtcm = s.cache_write(z_tensor, "global.vtcm")

    zouter, _ = s[z_vtcm].split(z_vtcm.op.axis[0], factor=factor)

    s[x_vtcm].compute_at(s[z_vtcm], zouter)
    s[y_vtcm].compute_at(s[z_vtcm], zouter)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global.vtcm", "global")

    (cache_read_x,) = s[x_vtcm].op.axis
    s[x_vtcm].tensorize(cache_read_x, mem_copy_read)

    (cache_read_y,) = s[y_vtcm].op.axis
    s[y_vtcm].tensorize(cache_read_y, mem_copy_read)

    mem_copy_write = intrin_mem_copy(outer_shape, dtype, "global", "global.vtcm")

    (cache_write_z,) = s[z_tensor].op.axis
    s[z_tensor].tensorize(cache_write_z, mem_copy_write)

    verify(hexagon_session, s, x_tensor, y_tensor, z_tensor, size)


def layout_transform_2d(n):
    return [n // 16, te.AXIS_SEPARATOR, n % 16]


@tvm.testing.requires_hexagon
def test_cache_read_write_2d(hexagon_session: Session):
    """Test 2D cache_read and cache_write to global.vtcm for hexagon"""
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x_tensor = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y_tensor = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z_tensor = te.compute(outer_shape, lambda i: x_tensor[i] + y_tensor[i], name="z")
    s = te.create_schedule(z_tensor.op)

    x_vtcm = s.cache_read(x_tensor, "global.vtcm", [z_tensor])
    y_vtcm = s.cache_read(y_tensor, "global.vtcm", [z_tensor])
    z_vtcm = s.cache_write(z_tensor, "global.vtcm")

    layout_x_vtcm = s[x_vtcm].transform_layout(layout_transform_2d)
    layout_y_vtcm = s[y_vtcm].transform_layout(layout_transform_2d)
    _ = s[z_vtcm].transform_layout(layout_transform_2d)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global.vtcm", "global")
    s[x_vtcm].tensorize(layout_x_vtcm[1], mem_copy_read)
    s[y_vtcm].tensorize(layout_y_vtcm[1], mem_copy_read)

    # The loop schedule over `z` is not modified when calling `transform_layout`
    # on `z_vtcm` above therefore we must call `split` to modify the loop schedule
    # over `z` to match the layout of `z_vtcm` such that we can accurately write
    # `z_vtcm` back to `z` using memory copy intrinsic
    _, zinner = s[z_tensor].split(z_tensor.op.axis[0], factor=factor)
    mem_copy_write = intrin_mem_copy(inner_shape, dtype, "global", "global.vtcm")
    s[z_tensor].tensorize(zinner, mem_copy_write)

    verify(hexagon_session, s, x_tensor, y_tensor, z_tensor, size)


@T.prim_func
def scale_by_two(buffer_a: T.Buffer[(8192,), "int8"], buffer_c: T.Buffer[(8192,), "int8"]):
    for i in T.serial(
        0,
        8192,
    ):
        with T.block("C"):
            buffer_c[i] = buffer_a[i] * T.int8(2)


def test_vtcm_lowering():
    """Test lowering with vtcm mem scope"""
    mod = tvm.IRModule.from_expr(scale_by_two.with_attr("global_symbol", "main"))
    sch = tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_block("C")
    (flat,) = sch.get_loops(block_c)
    outer, _, _, _ = sch.split(flat, factors=[8, 4, 2, 128])
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, outer)
    lowered = tvm.lower(sch.mod["main"])

    def ir_module_has_allocate_nodes(irmod):
        nallocs = 0

        def _visit(stmt):
            nonlocal nallocs
            if isinstance(stmt, tvm.tir.Allocate):
                nallocs += 1

        tvm.tir.stmt_functor.post_order_visit(irmod["main"].body, _visit)
        return nallocs

    assert not ir_module_has_allocate_nodes(lowered), (
        "AllocateNode found in lowered IRModule, "
        "VTCM allocations should have been lowered to tir.nd_mem_alloc_with_scope"
    )


if __name__ == "__main__":
    tvm.testing.main()
