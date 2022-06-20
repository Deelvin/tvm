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

import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from utils.adreno_utils import gpu_preprocess, build_run_compare


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 42, 42)
    filter_shape = (96, 32, 3, 3)
    bias_shape = (1, 96, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, [], gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape = (96, 32, 2, 2)
    bias_shape = (1, 96, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, [], gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_35_35_strides():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 48, 35, 35)
    filter_shape = (64, 48, 5, 5)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[2, 2, 2, 2],
        strides=[1, 1],
        out_dtype=dtype,
        channels=64,
        kernel_size=(5, 5),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, [], gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_resnet50_v2_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 7, 7)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        kernel_size=(7, 7),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 3, 299, 299)
    filter_shape = (64, 3, 3, 3)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_1x1_16c16spatial():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 16, 256, 256)
    filter_shape = (32, 16, 4, 4)
    bias_shape = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_4x4_16c16pad():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 256, 256)
    filter_shape = (32, 32, 4, 4)
    bias_shape = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[3, 3, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_4x4x4_16c16pad():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 256, 256)
    filter_shape = (4, 32, 4, 4)
    bias_shape = (1, 4, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[3, 3, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=4,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_yolov3_v2_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 1024, 13, 13)
    filter_shape = (255, 1024, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=255,
        kernel_size=(1, 1),
    )

    mod = relay.Function([A, B], conv)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params = {
        "weight": tvm.nd.array(filter_data),
    }

    build_run_compare(mod, params, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_2conv2d():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (96, 32, 2, 2)
    filter_shape2 = (32, 96, 2, 2)
    bias_shape1 = (1, 96, 1, 1)
    bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    B2 = relay.var("bias2", shape=bias_shape2, dtype=dtype)

    # C = relay.nn.relu(A)
    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv1, B1)
    D = relay.op.nn.relu(D)

    conv2 = relay.nn.conv2d(
        D,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv2, B2)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, W1, B1, W2, B2], D)
    print(mod)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    bias_data2 = np.zeros(bias_shape2).astype(dtype)
    initializer("weight", filter_data2)
    initializer("bias", bias_data2)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "bias2": tvm.nd.array(bias_data2),
    }

    static_memory_scope = [
        "global",
        "global",
        "global.texture-weight",
        "global.texture-weight",
        "global.texture-nhwc",
        "global.texture-weight",
        "global.texture-weight",
        "",
        "",
    ]

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)

@tvm.testing.requires_opencl
def test_residual_block():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    # bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    # B2 = relay.var("bias2", shape=bias_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv1, B1)
    D = relay.op.nn.relu(D)

    conv2 = relay.nn.conv2d(
        D,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )
    D = relay.op.add(conv2, D)
    D = D * relay.const(0.15, "float16")
    D = relay.op.nn.relu(D)

    conv3 = relay.nn.conv2d(
        D,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    D = relay.op.nn.relu(conv3)

    mod = relay.Function([A, W1, B1, W2, W3], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    # bias_data2 = np.zeros(bias_shape2).astype(dtype)
    initializer("weight", filter_data2)
    # initializer("bias", bias_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        # "bias2": tvm.nd.array(bias_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        "global",
        "global",
        "global.texture-weight",
        "global.texture-weight",
        "global.texture",
        "global.texture-weight",
        'global',
        "global.texture",
        "global.texture-weight",
        "",
        ""
    ]

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)


@tvm.testing.requires_opencl
def test_plan_device_issue1():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)

    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv2 = relay.nn.conv2d(
        conv1,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    mod = relay.Function([A, W1, W2], conv2)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    initializer("weight", filter_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "weight2": tvm.nd.array(filter_data2),
    }

    # static_memory_scope = [
    #     "global",
    #     "global",
    #     "global.texture-weight",
    #     "global.texture-weight",
    #     "global.texture-nhwc",
    #     "global.texture-weight",
    #     "global.texture-weight",
    #     "global",
    #     "global",
    # ]

    static_memory_scope = []

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)

@tvm.testing.requires_opencl
def test_branch_textures():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (96, 32, 2, 2)
    filter_shape2 = (32, 96, 2, 2)
    filter_shape3 = (5, 96, 2, 2)
    bias_shape1 = (1, 96, 1, 1)
    bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)
    B2 = relay.var("bias2", shape=bias_shape2, dtype=dtype)

    # C = relay.nn.relu(A)
    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv1, B1)
    D = relay.op.nn.relu(D)

    conv2 = relay.nn.conv2d(
        D,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv2 = relay.op.add(conv2, B2)
    conv2 = relay.op.nn.relu(conv2)

    conv3 = relay.nn.conv2d(
        D,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=5,
        kernel_size=(2, 2),
    )

    t = relay.Tuple([conv2, conv3])
    c = relay.op.concatenate(t, axis=1)


    mod = relay.Function([A, W1, B1, W2, B2, W3], c)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    bias_data2 = np.zeros(bias_shape2).astype(dtype)
    initializer("weight", filter_data2)
    initializer("bias", bias_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "bias2": tvm.nd.array(bias_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    # static_memory_scope = [
    #     "global",
    #     "global",
    #     "global.texture-weight",
    #     "global.texture-weight",
    #     "global.texture-nhwc",
    #     "global.texture-weight",
    #     "global.texture-weight",
    #     "global",
    #     "global",
    # ]

    static_memory_scope = []

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)


@tvm.testing.requires_opencl
def test_branch1_texture_params():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    # bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W0 = relay.var("weight0", shape=filter_shape0, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv0 = relay.nn.conv2d(
        A,
        W0,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    pool = relay.nn.avg_pool2d(conv0, pool_size=(2, 2), strides=(2, 2))
    conv1 = relay.nn.conv2d(
        pool,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv1 = relay.op.add(conv1, B1)
    conv1 = relay.op.nn.relu(conv1)

    conv2 = relay.nn.conv2d(
        pool,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        pool,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 1, 1, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv3 = relay.op.nn.relu(conv3)
    res = relay.op.add(conv1, conv2)
    res = relay.op.add(res, conv3)

    mod = relay.Function([A, W0, W1, B1, W2, W3], res)
    #print(mod)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data0 = np.zeros(filter_shape0).astype(dtype)
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight0": tvm.nd.array(filter_data0),
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        # "global",
        # "global",
        # "global.texture-weight",
        # "global.texture-weight",
        # "global.texture",
        # "global.texture-weight",
        # 'global',
        # "global.texture",
        # "global.texture-weight",
        # "",
        # ""
    ]

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)


#                      conv2d <- to get textures
#               /         \           \         <- here should be textures and textures in params
#          conv2d       conv2d       conv2d
#            \             /
#                  add                          <- tail required to have  the only one output
#                    \                /
#                          add
@tvm.testing.requires_opencl
def test_branch2_texture_params():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    # bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W0 = relay.var("weight0", shape=filter_shape0, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv0 = relay.nn.conv2d(
        A,
        W0,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv1 = relay.nn.conv2d(
        conv0,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv1 = relay.op.add(conv1, B1)
    conv1 = relay.op.nn.relu(conv1)

    conv2 = relay.nn.conv2d(
        conv0,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        conv0,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 1, 1, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv3 = relay.op.nn.relu(conv3)
    res = relay.op.add(conv1, conv2)
    res = relay.op.add(res, conv3)

    mod = relay.Function([A, W0, W1, B1, W2, W3], res)
    #print(mod)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data0 = np.zeros(filter_shape0).astype(dtype)
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight0": tvm.nd.array(filter_data0),
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        # "global",
        # "global",
        # "global.texture-weight",
        # "global.texture-weight",
        # "global.texture",
        # "global.texture-weight",
        # 'global',
        # "global.texture",
        # "global.texture-weight",
        # "",
        # ""
    ]

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)

# function repeat, params scope are different in reused functions
@tvm.testing.requires_opencl
def test_conv2d_different_param_scope():
    target = "opencl --device=adreno"
    dtype = "float16"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)

    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv2 = relay.nn.conv2d(
        conv1,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        conv2,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    mod = relay.Function([A, W1], conv3)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
    }

    static_memory_scope = [
        # "global",
        # "global",
        # "global.texture-weight",
        # "global.texture-weight",
        # "global.texture-nhwc",
        # "global.texture-weight",
        # "global.texture-weight",
        # "global",
        # "global",
    ]

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, static_memory_scope)