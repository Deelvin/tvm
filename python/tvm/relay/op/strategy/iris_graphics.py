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
"""Definition of Intel Iris Graphics operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_NCHWc_strategy.register("iris_graphics")
@conv2d_strategy.register("iris_graphics")
def conv2d_strategy_iris_graphics(attrs, inputs, out_type, target):
    """conv2d iris_graphics strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    # TODO(amalyshe): think if we can do next verification more gracious way
    if groups == 1:
        if (
            (data_layout == "NCHW" and kernel_layout == "OIHW")
            or (data_layout == "NCHW4c" and kernel_layout == "OIHW4o")
            or (data_layout == "NCHW" and kernel_layout == "OIHW4o")
            or (data_layout == "NCHW8c" and kernel_layout == "OIHW8o")
            or (data_layout == "NCHW" and kernel_layout == "OIHW8o")
        ):
            if len(kernel.shape) == 4:
                _, _, kh, kw = get_const_tuple(kernel.shape)
            else:
                _, _, kh, kw, _ = get_const_tuple(kernel.shape)
            # if (
            #     (2 < kh < 8 and 2 < kw < 8 and kh == kw)
            #     and (stride_h == 1 and stride_w == 1)
            #     and (dilation_h == 1 and dilation_w == 1)
            #     and not (data_layout == "NCHW" and (kernel_layout == "OIHW4o" or kernel_layout == "OIHW8o"))
            # ):
            #     strategy.add_implementation(
            #         wrap_compute_conv2d(topi.iris.conv2d_nchw_winograd_iris),
            #         wrap_topi_schedule(topi.adreno.schedule_conv2d_nchw_winograd),
            #         name="conv2d_nchw_winograd.iris",
            #         plevel=5,
            #     )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.iris_graphics.conv2d_nchwc_iris),
                wrap_topi_schedule(topi.iris_graphics.schedule_conv2d_nchwc),
                name="conv2d_nchwc.iris",
                plevel=10,
            )
        elif (
            (data_layout == "NHWC" and kernel_layout == "HWIO")
            or (data_layout == "NHWC4c" and kernel_layout == "HWIO4o")
            or (data_layout == "NHWC" and kernel_layout == "HWIO4o")
            or (data_layout == "NHWC8c" and kernel_layout == "HWIO8o")
            or (data_layout == "NHWC" and kernel_layout == "HWIO8o")
        ):
            if len(kernel.shape) == 4:
                kh, kw, _, _ = get_const_tuple(kernel.shape)
            else:
                kh, kw, _, _, _ = get_const_tuple(kernel.shape)
            # if (
            #     (2 < kh < 8 and 2 < kw < 8 and kh == kw)
            #     and (stride_h == 1 and stride_w == 1)
            #     and (dilation_h == 1 and dilation_w == 1)
            #     and not (data_layout == "NHWC" and (kernel_layout == "HWIO4o" or kernel_layout == "HWIO8o"))
            # ):
            #     strategy.add_implementation(
            #         wrap_compute_conv2d(topi.iris_graphics.conv2d_nhwc_winograd),
            #         wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc_winograd),
            #         name="conv2d_nhwc_winograd.iris",
            #         plevel=5,
            #     )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.iris_graphics.conv2d_nhwc_iris),
                wrap_topi_schedule(topi.iris_graphics.schedule_conv2d_nhwc_iris),
                name="conv2d_nhwc.iris",
                plevel=10,
            )
        else:
            raise RuntimeError(
                "Layout not supported: ("
                + data_layout
                + ", "
                + kernel_layout
                + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
            )
    else:
        # cannot use is_depthwise_conv2d because it does not know about NHWC4c/HWOI4o layouts
        if data_layout == "NCHW":
            ic = data.shape[1]
        elif data_layout == "NCHW4c" or data_layout == "NCHW8c": # TODO(amalyshe): better to generalize
            ic = data.shape[1] * data.shape[4]
        elif data_layout == "NHWC":
            ic = data.shape[3]
        elif data_layout == "NHWC4c" or data_layout == "NHWC8c": # TODO(amalyshe): better to generalize
            ic = data.shape[3] * data.shape[4]
        else:
            raise RuntimeError("Unsupported depthwise_conv2d data layout {}".format(data_layout))
        if kernel_layout == "OIHW":
            oc = kernel.shape[0]
        elif kernel_layout == "OIHW4o" or kernel_layout == "OIHW8o": # TODO(amalyshe): better to generalize
            oc = kernel.shape[0] * kernel.shape[4]
        elif kernel_layout == "HWOI":
            oc = kernel.shape[2]
        elif kernel_layout == "HWOI4o" or kernel_layout == "HWOI8o": # TODO(amalyshe): better to generalize
            oc = kernel.shape[2] * kernel.shape[4]
        else:
            raise RuntimeError(
                "Unsupported depthwise_conv2d kernel layout {}".format(kernel_layout)
            )

        if ic == oc == groups:
            if ((data_layout == "NCHW" and kernel_layout == "OIHW")
                or (data_layout == "NCHW4c" and kernel_layout == "OIHW4o")
                or (data_layout == "NCHW8c" and kernel_layout == "OIHW8o") # TODO(amalyshe): better to generalize
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.iris_graphics.depthwise_conv2d_nchwc_iris),
                    wrap_topi_schedule(topi.iris_graphics.schedule_depthwise_conv2d_nchwc),
                    name="depthwise_conv2d_nchwc.iris",
                    plevel=10,
                )
            elif ((data_layout == "NHWC" and kernel_layout == "HWOI")
                   or (data_layout == "NHWC4c" and kernel_layout == "HWOI4o") # TODO(amalyshe): better to generalize
                   or (data_layout == "NHWC8c" and kernel_layout == "HWOI8o")
            ):
                if data.shape[-1] >= 4:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.iris_graphics.depthwise_conv2d_nhwc),
                        wrap_topi_schedule(topi.iris_graphics.schedule_depthwise_conv2d_nhwc),
                        name="depthwise_conv2d_nhwc.iris",
                        plevel=10,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                        wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                        name="depthwise_conv2d_nhwc.cuda",
                    )
            else:
                raise RuntimeError(
                    "Layout not supported: ("
                    + data_layout
                    + ", "
                    + kernel_layout
                    + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
                )
        else:
            raise RuntimeError("General group convolution is not currently supported")
    return strategy


@conv2d_winograd_without_weight_transform_strategy.register("iris_graphics")
def conv2d_winograd_without_weight_transform_strategy_iris_graphics(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transform iris_graphics strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not support arbitrary group number"
    strategy = _op.OpStrategy()
    if layout in ("NCHW", "NCHW4c", "NCHW8c"): # TODO(amalyshe): this form is good for generalization
        strategy.add_implementation(
            wrap_compute_conv2d(topi.iris_graphics.conv2d_nchw_winograd_without_weight_transform_iris),
            wrap_topi_schedule(topi.iris_graphics.schedule_conv2d_nchw_winograd_without_weight_transform),
            name="conv2d_nchw_winograd_without_weight_transform.iris",
            plevel=5,
        )
    elif layout in ("NHWC", "NHWC4c", "NHWC8c"):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.iris_graphics.conv2d_nhwc_winograd_without_weight_transform),
            wrap_topi_schedule(topi.iris_graphics.schedule_conv2d_nhwc_winograd_without_weight_transform),
            name="conv2d_nhwc_winograd_without_weight_transform.iris",
            plevel=5,
        )
    else:
        raise RuntimeError(
            "Unsupported conv2d_winograd_without_weight_transform layout {}".format(layout)
        )
    return strategy


@schedule_pool.register("iris_graphics")
def schedule_pool_iris_graphics(attrs, outs, target):
    """schedule pooling ops for iris_graphics"""
    with target:
        if attrs.layout  in ("NCHW4c", "NCHW8c"):
            return topi.adreno.schedule_pool(outs, attrs.layout)
        return topi.cuda.schedule_pool(outs, attrs.layout)


@schedule_injective.register(["iris_graphics"])
def schedule_injective_iris_graphics(attrs, outs, target):
    """schedule injective ops for iris_graphics"""
    with target:
        return topi.adreno.schedule_injective(outs)


@schedule_reduce.register(["iris_graphics"])
def schedule_reduce_iris_graphics(attrs, outs, target):
    """schedule reduction ops for iris_graphics GPU"""
    with target:
        return topi.adreno.schedule_reduce(outs)


@schedule_adaptive_pool.register(["iris_graphics"])
def schedule_adaptive_pool_iris_graphics(attrs, outs, target):
    """schedule adaptive pooling ops for iris_graphics"""
    with target:
        return topi.adreno.schedule_adaptive_pool(outs, attrs.layout)


@concatenate_strategy.register(["iris_graphics"])
def concatenate_strategy_iris_graphics(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_concat(topi.transform.concatenate),
        wrap_topi_schedule(topi.adreno.schedule_injective),
        name="concatenate.adreno",
    )
    return strategy
