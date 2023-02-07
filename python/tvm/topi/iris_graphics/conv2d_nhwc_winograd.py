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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Winograd NHWC template for Intel Iris Graphics backend"""
from ..adreno.conv2d_nhwc_winograd import conv2d_nhwc_winograd_comp, schedule_conv2d_winograd_impl
from tvm import autotvm

@autotvm.register_topi_schedule("conv2d_nhwc_winograd.iris")
def schedule_conv2d_nhwc_winograd(cfg, outs):
    return schedule_conv2d_winograd_impl(cfg, outs, tag="dummy_compute_at")

@autotvm.register_topi_compute("conv2d_nhwc_winograd.iris")
def conv2d_nhwc_winograd_iris(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return conv2d_nhwc_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=False, default_block_size = 4
    )

@autotvm.register_topi_schedule("conv2d_nhwc_winograd_without_weight_transform.iris")
def schedule_conv2d_nhwc_winograd_without_weight_transform(cfg, outs):
    return schedule_conv2d_winograd_impl(cfg, outs, tag="dummy_compute_at", pre_computed=True)

@autotvm.register_topi_compute("conv2d_nhwc_winograd_without_weight_transform.iris")
def conv2d_nhwc_winograd_without_weight_transform_iris(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    return conv2d_nhwc_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=True, default_block_size = 4
    )