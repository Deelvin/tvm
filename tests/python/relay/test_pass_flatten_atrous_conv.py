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
# pylint: disable=unused-wildcard-import
from ast import arg
import numpy as np
import pytest
from torch import float32
import tvm
from tvm import relay
# from tvm.relay.transform import fake_quantization_to_integer
# from tvm.contrib import graph_executor


def test_s2b_conv2d_b2s():
    # The single-argument operation is converted.
    shape_x = [1, 65, 65, 384]# [1, 2, 4]
    op78 = relay.var("p0", shape=shape_x, dtype="float32")
    # a = relay.var("a", shape=shape_x, dtype="int8")

    # op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))

    # op1 = relay.op.reshape(op0, (1, 4, 2))
    # expr = relay.op.erf(op1)


        # x = relay.var("x", shape=(1, 56, 56, 64))
        # weight = relay.var("weight", shape=(3, 3, 64, 64))
        # y = relay.nn.conv2d(
        #     x,
        #     weight,
        #     channels=64,
        #     kernel_size=(3, 3),
        #     padding=(1, 1),
        #     data_layout="NHWC",
        #     kernel_layout="HWIO",
        # )

#   op78 =  /* ty=Tensor[(1, 65, 65, 384), float32] */
    # meta_const_51 = relay.const(np.random.uniform(size=[3, 3, 384, 1]).astype("float32")) # /* ty=Tensor[(3, 3, 384, 1), float32] */  
    meta_const_51 = relay.var("meta_const_51", shape=[3, 3, 384, 1], dtype="float32")

    # op79 = relay.op.erf(op78)
    op79 = relay.nn.space_to_batch_nd(op78, block_shape=[2, 2], paddings=[[2, 3], [2, 3]]) # /* ty=Tensor[(4, 35, 35, 384), float32] */;
    op80 = relay.nn.conv2d(op79, meta_const_51, padding=[0, 0, 0, 0], groups=384, channels=384, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    op81 = relay.nn.batch_to_space_nd(op80, block_shape=[2, 2], crops=[[0, 1], [0, 1]]) #/* ty=Tensor[(1, 65, 65, 384), float32] */;
    expr = op81
    # expr = op79
    # mod_exp = tvm.parser.fromtext(
    #     """
    #     #[version = "0.0.5"]
    #     def @main(%p0: Tensor[(1, 56, 56, 128), int16], %p1: Tensor[(3, 3, 128, 1), int16], %p2: Tensor[(1, 1, 1, 128), int32]){
    #       %0 = nn.conv2d(%p0, %p1, padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI", out_dtype="int32") /* ty=Tensor[(1, 56, 56, 128), int32] */;
    #       %1 = add(%0, %p2) /* ty=Tensor[(1, 56, 56, 128), int32] */;
    #       %2 = fixed_point_multiply(%1, multiplier=2080045879, shift=-4) /* ty=Tensor[(1, 56, 56, 128), int32] */;
    #       %3 = clip(%2, a_min=0f, a_max=255f) /* ty=Tensor[(1, 56, 56, 128), int32] */;
    #       cast(%3, dtype="uint8") /* ty=Tensor[(1, 56, 56, 128), uint8] */
    #     }
    #     """
    # )

    # mod_exp = tvm.parser.fromtext(
    #     """
    #     #[version = "0.0.5"]
    #     def @main(%p0: Tensor[(1, 65, 65, 384), float32]) -> Tensor[(1, 65, 65, 384), float32] {
    #         %0 = nn.space_to_batch_nd(%p0, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    #         %1 = nn.conv2d(%0, %p0, padding=[0, 0, 0, 0], groups=384, channels=384, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI")
    #         nn.batch_to_space_nd(%1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
    #     }
    # """)

    # op0 = relay.op.reshape(a, (1, 4, 2))
    # op1 = relay.qnn.op.dequantize(op0, relay.const(2.0), relay.const(0))
    # expected_expr = relay.op.erf(op1)
    x_np = np.random.random_sample(shape_x).astype("float32")
    # x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=[3, 3, 384, 1], dtype="int8").astype("float32")

    
    # print(x_np)
    print(x_np.shape)
    # args = [x_np]
    args = [x_np, w_np]
    mod = tvm.IRModule.from_expr(expr)
    # mod_def = tvm.relay.transform.InferType()(mod)
    mod_def = tvm.relay.transform.InferType()(mod)
    # mod_int = tvm.relay.transform.FakeQuantizationToInteger(False, False)(mod_def)
    # mod_int = mod_def
    mod_int = tvm.relay.transform.FlattenAtrousConv()(mod_def)
    # mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    print("mod\n", mod)
    print("mod_def\n", mod_def)
    print("mod_int\n", mod_int)
    # print("mod_exp\n", mod_exp)
    assert not tvm.ir.structural_equal(mod, mod_int)
    # assert tvm.ir.structural_equal(mod_int, mod_exp)
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    # result_int = (
    #     relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
    #     .evaluate()(*args)
    #     .numpy()
    # )
    # result_exp = (
    #     relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
    #     .evaluate()(*args)
    #     .numpy()
    # )
    # if False:
    #     assert np.all(np.abs(result_def.astype("int32") - result_int.astype("int32")) <= 1)
    # else:
    #     assert np.array_equal(result_def, result_int)

    # assert np.array_equal(result_int, result_exp)




def test_fq_positive_single_arg_part_ICEMIST():
    # The single-argument operation is converted.
    shape_x = [1, 2, 4]
    a = relay.var("a", shape=shape_x, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))

    op1 = relay.op.reshape(op0, (1, 4, 2))
    expr = relay.op.erf(op1)

    op0 = relay.op.reshape(a, (1, 4, 2))
    op1 = relay.qnn.op.dequantize(op0, relay.const(2.0), relay.const(0))
    expected_expr = relay.op.erf(op1)
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    # compare_expected_fq_qat_to_int(expr, expected_expr, )
    args = [x_np]
    allow_rounding_error=False

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False, True)(mod_def)
    mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    print("mod_def\n", mod_def)
    print("mod_int\n", mod_int)
    print("mod_exp\n", mod_exp)
    assert not tvm.ir.structural_equal(mod, mod_int)
    assert tvm.ir.structural_equal(mod_int, mod_exp)
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_exp = (
        relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    if allow_rounding_error:
        assert np.all(np.abs(result_def.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result_def, result_int)

    assert np.array_equal(result_int, result_exp)


def test_fq_positive_single_arg_part_ICEMIS2T():
    # dtype = "int8"
    # a = relay.var("a", shape=[3], dtype=dtype)
    # expr = relay.op.add(a, relay.const(1,dtype=dtype))

    # a = relay.var("a", shape=[3], dtype="int8")
    # expr = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    # # expr = relay.op.erf(a)
    # x_np = np.array([1.0, 2.0, 3.0]).astype("float32")

    dtype="float32"
    shape_x = [3]
    a = relay.var("a", shape=shape_x, dtype=dtype)
    op1 = relay.op.add(a, relay.const(1,dtype=dtype))
    expr = op1
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype(dtype)

    args = [x_np]

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)

    target = "llvm"
    device = tvm.cpu()

    result_def = (
        relay.create_executor("vm", mod=mod_def, device=device, target=target)
        .evaluate()(*args)
        .numpy()
    )

    # result_def = (
    #     relay.create_executor("graph", mod=mod_def, device=device, target=target)
    #     .evaluate()(*args)
    #     .numpy()
    # )

    print(result_def)


def test_s2b_conv2d_b2s_ice__():
    # The single-argument operation is converted.
    np.random.seed(1)

    shape_x = [1, 5, 5, 4]#[1, 65, 65, 384]
    groups=4#384
    channels=4#384
    # shape_x = [1, 65, 65, 384]
    op78 = relay.var("p0", shape=shape_x, dtype="float32")
    meta_const_51 = relay.var("meta_const_51", shape=[3, 3, groups, 1], dtype="float32")


    op79 = relay.nn.space_to_batch_nd(op78, block_shape=[2, 2], paddings=[[2, 3], [2, 3]]) # /* ty=Tensor[(4, 35, 35, 384), float32] */;
    op80 = relay.nn.conv2d(op79, meta_const_51, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    op81 = relay.nn.batch_to_space_nd(op80, block_shape=[2, 2], crops=[[0, 1], [0, 1]]) #/* ty=Tensor[(1, 65, 65, 384), float32] */;
    expr = op81
  
    op80 = relay.nn.conv2d(op78, meta_const_51, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    expected_expr = op80

    # x_np = np.random.random_sample(shape_x).astype("float32")
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=[3, 3, groups, 1], dtype="int8").astype("float32")

    
    args = [x_np, w_np]
    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    # mod_int = tvm.relay.transform.FlattenAtrousConv()(mod_def)
    mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    print("mod\n", mod)
    print("mod_def\n", mod_def)
    # print("mod_int\n", mod_int)
    print("mod_exp\n", mod_exp)
    # assert not tvm.ir.structural_equal(mod, mod_int)
    # assert tvm.ir.structural_equal(mod_int, mod_exp)
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    # result_int = (
    #     relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
    #     .evaluate()(*args)
    #     .numpy()
    # )
    result_exp = (
        relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )

    # assert np.array_equal(result_def, result_int)
    np.set_printoptions(threshold=sys.maxsize)
    with open("result_def.txt", "w") as f: f.write(str(result_def))
    with open("result_exp.txt", "w") as f: f.write(str(result_exp))
    assert np.array_equal(result_def, result_exp)

    # assert np.array_equal(result_int, result_exp)

def test_s2b_conv2d_b2s_ice():
    # The single-argument operation is converted.
    np.random.seed(1)


    # shape_x = [1, 65, 65, 384]
    # groups=384
    # channels=384

    shape_x = [1, 5, 5, 4]
    groups=4
    channels=4

    shape_w = [3, 3, groups, 1]

    op78 = relay.var("p0", shape=shape_x, dtype="float32")
    w_np = np.full(shape_w, 1).astype("float32")
    # w_np = np.full(shape_w, 1).astype("float32")
    # w_np = np.arange(shape_w[0] * shape_w[1] * shape_w[2] * shape_w[3]).reshape(shape_w).astype("float32")
    meta_const_51 = relay.const(w_np) # /* ty=Tensor[(3, 3, 384, 1), float32] */  
    # meta_const_51 = relay.var("meta_const_51", shape=shape_w, dtype="float32")
    
    
    op79 = relay.nn.space_to_batch_nd(op78, block_shape=[2, 2], paddings=[[2, 3], [2, 3]]) # /* ty=Tensor[(4, 35, 35, 384), float32] */;
    op80 = relay.nn.conv2d(op79, meta_const_51, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    op81 = relay.nn.batch_to_space_nd(op80, block_shape=[2, 2], crops=[[0, 1], [0, 1]]) #/* ty=Tensor[(1, 65, 65, 384), float32] */;
    expr = op81

    # expr = op81
  
    # op80 = relay.nn.conv2d(op78, meta_const_51, padding=[0, 2, 2, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    # # op80 = relay.nn.conv2d(op78, meta_const_51, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    # expected_expr = op80

    # op80 = relay.nn.conv2d(op78, meta_const_51, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    # expected_expr_2 = op79

    # x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    x_np = np.arange(shape_x[0] * shape_x[1] * shape_x[2] * shape_x[3]).reshape(shape_x).astype("float32")
    # x_np = np.full(shape_x, 2).astype("float32")

    # w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")
    # w_np = np.arange(shape_w[0] * shape_w[1] * shape_w[2] * shape_w[3]).reshape(shape_w).astype("float32")
    # w_np = np.arange(shape_w[0] * shape_w[1] * shape_w[2] * shape_w[3]).reshape(shape_w).astype("float32")
    # w_np = np.full(shape_w, 1).astype("float32")
    # shape_w_sp = [5,5,4,1]
    # w_np_sp = np.full(shape_w_sp, 1).astype("float32")


    print("######################################################")
    print("x_np", x_np.shape)
    print(repr(x_np))
    print("w_np", w_np.shape)
    print(repr(w_np))
    print("######################################################")
    


    args = [x_np]
    # args = [x_np, w_np]
    # args_sp = [x_np, w_np_sp]
    # args_2 = [x_np, w_np]
    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    # is_tf = True
    is_tf = False
    mod_int = tvm.relay.transform.FlattenAtrousConv(is_tf)(mod_def)
    # mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    # mod_exp_2 = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr_2))
    print("mod\n", mod)
    print("mod_def\n", mod_def)
    print("mod_int\n", mod_int)
    # print("mod_exp\n", mod_exp)
    # print("mod_exp\n", mod_exp_2)
    # assert not tvm.ir.structural_equal(mod, mod_int)
    # assert tvm.ir.structural_equal(mod_int, mod_exp)
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        # .evaluate()(*args_sp)
        .evaluate()(*args)
        .numpy()
    )
    # result_exp = (
    #     relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
    #     .evaluate()(*args)
    #     .numpy()
    # )

    # result_exp_2 = (
    #     relay.create_executor("vm", mod=mod_exp_2, device=tvm.cpu(), target="llvm")
    #     .evaluate()(*args_2)
    #     .numpy()
    # )

    # assert np.array_equal(result_def, result_int)
    np.set_printoptions(threshold=sys.maxsize)
    with open("result_def.txt", "w") as f: 
        f.write(str(result_def.shape) + "\n")
        f.write(str(result_def))
    with open("result_int.txt", "w") as f: 
        f.write(str(result_int.shape) + "\n")
        f.write(str(result_int))
    

    print("######################################################")
    print("result_def", result_def.shape)
    print(repr(result_def))
    print("result_int", result_int.shape)
    print(repr(result_int))
    # print("result_exp", result_exp.shape)
    # print(result_exp)
    print("######################################################")
    assert np.array_equal(result_def, result_int)

    # assert np.array_equal(result_int, result_exp)



def test_s2b_conv2d_b2s_ice_2():
    # The single-argument operation is converted.
    np.random.seed(1)


    # shape_x = [1, 65, 65, 384]
    # groups=384
    # channels=384

    shape_x = [1, 5, 5, 4]
    groups=4
    channels=4

    shape_w = [3, 3, groups, 1]

    op78 = relay.var("p0", shape=shape_x, dtype="float32")
    w_np = np.full(shape_w, 1).astype("float32")
    meta_const_51 = relay.const(w_np) # /* ty=Tensor[(3, 3, 4, 1), float32] */


    shape_w_2 = [5, 5, groups, 1]
    w_np_2 = np.full(shape_w_2, 1).astype("float32")
    meta_const_51_2 = relay.const(w_np_2)





    op79 = relay.nn.space_to_batch_nd(op78, block_shape=[2, 2], paddings=[[2, 3], [2, 3]]) # /* ty=Tensor[(4, 5, 5, 4), float32] */;
    op80 = relay.nn.conv2d(op79, meta_const_51 , padding=[0, 0, 0, 0], 
        groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 3, 3, 4), float32] */;
    op81 = relay.nn.batch_to_space_nd(op80, block_shape=[2, 2], crops=[[0, 1], [0, 1]]) # /* ty=Tensor[(1, 5, 5, 4), float32] */
    expr = op81 # Tensor[(1, 5, 5, 4), float32]
    # expr = op80 
    # expr = op81 
    # expr = op79 
  
    expected_expr = relay.nn.conv2d(op78, meta_const_51, padding=[2, 2, 2, 2], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;
    # expected_expr = relay.nn.conv2d(op78, meta_const_51_2, padding=[0, 0, 0, 0], groups=groups, channels=channels, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI") # /* ty=Tensor[(4, 33, 33, 384), float32] */;



    x_np = np.full(shape_x, 2).astype("float32")


    print("######################################################")
    print("x_np", x_np.shape)
    print(x_np)
    print("w_np", w_np.shape)
    print(w_np)
    # print("w_np_2", w_np_2.shape)
    # print(w_np_2)
    print("######################################################")
    


    args = [x_np]

    mod_def = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expr))
    mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    print("mod_def\n", mod_def)
    print("mod_exp\n", mod_exp)

    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_exp = (
        relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    

    print("######################################################")
    print("result_def", result_def.shape)
    print(result_def)
    # print("result_exp", result_exp.shape)
    # print(result_exp)
    print("######################################################")
    assert np.array_equal(result_def, result_exp)

    # assert np.array_equal(result_int, result_exp)
if __name__ == "__main__":
    import sys
    # test_s2b_conv2d_b2s_ice_2()
    test_s2b_conv2d_b2s_ice()
    # test_fq_positive_single_arg_part_ICEMIS2T()
    # test_s2b_conv2d_b2s_ice()
    # test_fq_positive_single_arg_part_ICEMIS2T()
    # test_fq_positive_single_arg_part_ICEMIST()
    # test_fq_qat_intermediate_infertype()
    # test_fq_positive_single_arg_part()
    # sys.exit(pytest.main([__file__] + sys.argv[1:]))
