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
"""Definition of avec operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_NCHWc_strategy.register("avec")
@conv2d_strategy.register("avec")
def conv2d_strategy_avec(attrs, inputs, out_type, target):
    """conv2d avec strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if (data_layout == "NCHW" and kernel_layout == "OIHW") or (
            data_layout == "NCHW4c" and kernel_layout == "OIHW4o"
        ):
            if len(kernel.shape) == 4:
                _, _, kh, kw = get_const_tuple(kernel.shape)
            else:
                _, _, kh, kw, _ = get_const_tuple(kernel.shape)
            if out_type.dtype == "float16":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.avec.conv2d_nchwc),
                    wrap_topi_schedule(topi.avec.schedule_conv2d_nchwc),
                    name="conv2d_nchwc.avec",
                    plevel=10,
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.avec.conv2d_nchwc_acc32),
                wrap_topi_schedule(topi.avec.schedule_conv2d_nchwc_acc32),
                name="conv2d_nchwc_acc32.avec",
                plevel=20,
            )
        else:
            raise RuntimeError(
                "Layout not supported: ("
                + data_layout
                + ", "
                + kernel_layout
                + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
            )
    return strategy

