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

import re
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare


dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nchw4c(remote, target, dtype):
    """Verification of the case NCHW->NCHW4c"""
    input_shape = (1, 32, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NCHW", "NCHW4c")
    mod = relay.Function([A], lt)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nchw(remote, target, dtype):
    """Verification of the case NCHW4c->NCHW"""
    input_shape = (1, 36, 1, 1, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NCHW4c", "NCHW")
    mod = relay.Function([A], lt)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nhwc4c(remote, target, dtype):
    """Verification of the case NHWC->NHWC4c"""
    input_shape = (1, 1, 1, 144)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NHWC", "NHWC4c")
    mod = relay.Function([A], lt)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nhwc(remote, target, dtype):
    """Verification of the case NHWC4c->NHWC"""
    input_shape = (1, 1, 1, 144)
    filter_shape = (1, 1, 144, 144)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W = relay.var("weight", shape=filter_shape, dtype=dtype)
    conv1 = relay.nn.conv2d(
        A,
        W,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=144,
        kernel_size=(1, 1),
    )

    #lt = relay.layout_transform(conv1, "NHWC4c", "NHWC")
    mod = relay.Function([A, W], conv1)

    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params = {
        "weight": tvm.nd.array(filter_data),
    }


    build_run_compare(remote, mod, params, {"data": input_shape}, {"data": dtype}, target)

if __name__ == "__main__":
    test_layout_transform_to_block_nhwc(None, "opencl -device=adreno", "float16")
