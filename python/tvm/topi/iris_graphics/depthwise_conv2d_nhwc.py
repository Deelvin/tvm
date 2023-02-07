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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""depthwise_conv2d_nchw(c) compute for Intel Iris Graphics"""
from ..adreno.depthwise_conv2d_nhwc import compute_depthwise_conv2d_nhwc, schedule_depthwise_conv2d_NHWC_HWOI
from tvm import te
from tvm import autotvm
from ..utils import traverse_inline

@autotvm.register_topi_schedule("depthwise_conv2d_nhwc.iris")
def schedule_depthwise_conv2d_nhwc(cfg, outs):
    """Create the schedule for depthwise conv2d_nchw4c_ohwi4o"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "adreno_dw_conv2d_latest_op":
            schedule_depthwise_conv2d_NHWC_HWOI(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute("depthwise_conv2d_nhwc.iris")
def depthwise_conv2d_nhwc(cfg, Input, Filter, stride, padding, dilation, out_dtype):
    compute_depthwise_conv2d_nhwc(cfg, Input, Filter, stride, padding, dilation, out_dtype, 8)