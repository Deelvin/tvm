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
import numpy as np
import pytest
import tvm
from tvm import relay


def compare_fq_to_int(expr, args, allow_rounding_error=False):
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)

    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )

    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)


def test_fake_quantize_conv():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
            kernel_size=[5, 5],
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_conv_per_channel():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
        one = relay.const([1.0] * 16)
        zero = relay.const([0] * 16)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0)),
            relay.qnn.op.dequantize(
                w, relay.const(np.random.random([16]).astype("float32")), zero, axis=0
            ),
            kernel_size=[5, 5],
            channels=16,
        )
        op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(0), out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np], allow_rounding_error=True)


def test_fake_quantize_transposeconv():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[3, 16, 5, 5], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.conv2d_transpose(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
            kernel_size=[5, 5],
            data_layout="NCHW",
            kernel_layout="IOHW",
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[3, 16, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_dense():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[128, 64], dtype="int8")
        w = relay.var("w", shape=[256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.dense(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_dense_per_channel():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[128, 64], dtype="int8")
        w = relay.var("w", shape=[256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.dense(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(
                w,
                relay.const(np.random.random([256]).astype("float32")),
                relay.const([0] * 256),
                axis=0,
            ),
            units=256,
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np], allow_rounding_error=True)


def test_fake_quantize_batch_matmul():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 128, 64], dtype="int8")
        w = relay.var("w", shape=[1, 256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.batch_matmul(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[1, 256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_transpose_quantize_conv():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

    compare_fq_to_int(op, [x_np, w_np])


def test_fake_transpose_quantize_conv_bias_add():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(op, relay.qnn.op.dequantize(bias, one, zero))
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np])


def test_fake_transpose_quantize_conv_bias_add_per_channel():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)
    w_scale = (np.random.random([16]).astype("float32") - 0.5) / 10 + 0.5
    noise = (np.random.random([16]).astype("float32") - 0.5) * 1e-15
    w_zp = relay.const([0] * 16)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(w_scale), w_zp, axis=0), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(
        op, relay.qnn.op.dequantize(bias, relay.const(2.0 * w_scale + noise), w_zp, axis=0)
    )
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np], allow_rounding_error=True)


def test_fake_transpose_quantize_conv_bias_add_mismatch():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    two = relay.const(2.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(op, relay.qnn.op.dequantize(bias, two, zero))
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np])


def test_fake_quantize_maxpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.max_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_avgpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.avg_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_global_avg_pool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.global_avg_pool2d(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_rsqrt():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.rsqrt(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.reshape(x, [1, 3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_expand_dims():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.expand_dims(x, axis=1)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_squeeze():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.squeeze(x, axis=[0])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_strided_slice():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.strided_slice(x, begin=[0, 0, 0, 0], end=[1, 1, 112, 112])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_split():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.split(x, axis=3, indices_or_sections=2)
    op = relay.qnn.op.quantize(op[0], relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])

    op = relay.op.split(x, axis=3, indices_or_sections=[56, 112, 168])
    op = relay.qnn.op.quantize(op[1], relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_batch_flatten():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.batch_flatten(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_transpose_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.transpose(x, [1, 0, 2, 3])
    op = relay.op.reshape(op, [3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_concat():
    zero = relay.const(0)
    inputs = []
    for i in range(4):
        inputs.append(
            relay.qnn.op.dequantize(
                relay.var("x%d" % i, shape=[1, 4], dtype="int8"), relay.const(i + 0.5), zero
            )
        )
    concat = relay.op.concatenate(inputs, axis=1)
    out = relay.qnn.op.quantize(concat, relay.const(3.5), zero)

    inputs_np = []
    for i in range(4):
        inputs_np.append(np.random.randint(-128, 127, size=[1, 4], dtype="int8"))

    compare_fq_to_int(out, inputs_np)


def test_fake_quantize_clip():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.clip(x, 0, 6)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_clip_per_channel():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(
        x, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), axis=1
    )
    op = relay.op.clip(x, 0, 6)
    op = relay.qnn.op.quantize(
        op, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), out_dtype="uint8", axis=1
    )

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_relu():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.nn.relu(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_relu_per_channel():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(
        x, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), axis=1
    )
    op = relay.op.nn.relu(x)
    op = relay.qnn.op.quantize(
        op, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), out_dtype="uint8", axis=1
    )

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


@pytest.mark.parametrize(
    "operator",
    [relay.op.add, relay.op.multiply, relay.op.subtract, relay.op.minimum, relay.op.maximum],
)
def test_fake_quantize_binary(operator):
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(0.1), relay.const(0))

    y = relay.var("y", shape=[1, 3, 224, 224], dtype="int8")
    y = relay.qnn.op.dequantize(y, relay.const(0.2), relay.const(0))

    op = operator(x, y)
    if operator == relay.op.multiply:
        out_scale = relay.const(20.0)
    else:
        out_scale = relay.const(0.1)

    op = relay.qnn.op.quantize(op, out_scale, relay.const(0), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")
    y_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np, y_np])


@pytest.mark.parametrize(
    "operator",
    [
        relay.op.add,
        relay.op.multiply,
        relay.op.subtract,
        relay.op.subtract,
        relay.op.minimum,
        relay.op.maximum,
    ],
)
def test_fake_quantize_binary_const(operator):
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(0.1), relay.const(10))

    y = relay.const(1.0)

    op = operator(x, y)
    op = relay.qnn.op.quantize(op, relay.const(0.1), relay.const(10), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_pad():
    x = relay.var("x", shape=[1, 383, 128], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(1.0), relay.const(10))
    op = relay.op.nn.pad(x, [[0, 0], [0, 1], [0, 0]], 0.0)
    op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(10), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 383, 128], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_depth_to_space():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.depth_to_space(x, 4)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_max_min():
    def run_test_case(partial_func):
        x = relay.var("x", shape=[1, 3, 10, 10], dtype="int8")

        zero = relay.const(0)
        x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
        # To be a little more realistic since max/min will rarely be by themselves
        x = relay.op.nn.depth_to_space(x, 4)
        op = partial_func(x)
        op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

        x_np = np.random.randint(-128, 127, size=[1, 3, 10, 10], dtype="int8")
        compare_fq_to_int(op, [x_np])

    run_test_case(relay.op.max)
    run_test_case(relay.op.min)

    # Test forwarding kwargs works
    run_test_case(lambda x: relay.op.max(x, axis=1))
    run_test_case(lambda x: relay.op.min(x, axis=1))


def test_fq_hard_fail():
    @tvm.ir.register_op_attr("nn.conv2d", "FTVMFakeQuantizationToInteger", level=11)
    def conv2d(expr, type_map):  # pylint: disable=unused-variable
        raise NotImplementedError

    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    op = relay.op.nn.conv2d(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        kernel_size=[5, 5],
    )
    op = relay.qnn.op.quantize(op, one, zero, out_dtype="int8")
    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    mod_int = tvm.relay.transform.FakeQuantizationToInteger(hard_fail=False)(mod)
    assert tvm.ir.structural_equal(mod_int, mod)
    # Catch a generic exception because the tvm FFI eats the python exception type
    with pytest.raises(Exception):
        mod_int = tvm.relay.transform.FakeQuantizationToInteger(hard_fail=True)(mod)


def test_dequantize_propagation():
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    # shape_w = [1, 8, 2]
    x = relay.var("x", shape=shape_x, dtype="int8")
    w = relay.var("w", shape=shape_w, dtype="int8")

    # a = relay.qnn.op.dequantize(x, relay.const(1.5), relay.const(0)) # input, scale, shift
    # b = relay.qnn.op.dequantize(w, relay.const(0.5), relay.const(0)) # input, scale, shift
    # op = relay.op.nn.batch_matmul(a, b)
    # op = relay.op.add(op, relay.const(2.0, "float32"))
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type


    # a = relay.qnn.op.dequantize(x, relay.const(1.5), relay.const(0)) # input, scale, shift
    # b = relay.qnn.op.dequantize(w, relay.const(0.5), relay.const(0)) # input, scale, shift
    # op = relay.op.nn.batch_matmul(a, b)
    # op = relay.op.add(a, b)
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.op.subtract(op, relay.const(1, "int8"))

    # op = relay.op.erf(op) # HERE
    # op = relay.op.multiply(op, relay.const(3.0))
    # op = relay.op.subtract(op, relay.const(1, "int8"))
    
    a = x
    b = w

    # a = relay.op.abs(x)
    # b = relay.op.abs(w)

    # a = relay.op.add(a, relay.const(2.0, "int8"))
    # b = relay.op.add(b, relay.const(3.0, "int8"))

    a = relay.qnn.op.dequantize(a, relay.const(1.5), relay.const(0)) # input, scale, shift
    b = relay.qnn.op.dequantize(b, relay.const(0.5), relay.const(0)) # input, scale, shift

    # op = relay.op.add(a, b)

    # op1 = relay.op.nn.batch_matmul(a, b)
    # op2 = relay.op.nn.batch_matmul(b, a)
    # op11 = relay.op.nn.batch_matmul(op1, op2)
    # op22 = relay.op.nn.batch_matmul(op2, op1)
    # op = relay.op.nn.batch_matmul(op11, op22)

    

    op = relay.op.nn.batch_matmul(a, b)
    # op2 = relay.op.nn.batch_matmul(b, a)
    # op = relay.op.nn.batch_matmul(op, op2)
    op = relay.op.add(op, relay.const(2.0, "float32"))

    op = relay.op.erf(op) # here
    # op = relay.op.multiply(op, relay.const(3.0))
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.op.subtract(op, relay.const(1, "int8"))

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")

    expr = op
    args = [x_np, w_np]
    allow_rounding_error=False

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False)(mod_def)
    print("mod tvm.relay.transform.InferType\n", mod_def, "\n")
    print("mod tvm.relay.transform.FakeQuantizationToInteger\n", mod_int, "\n")
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    print("result.astype(int32)", result.astype("int32"))
    print("result_int.astype(int32)", result_int.astype("int32"))
    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)


def test_dequantize_propagation_test():
    shape_x = [1, 1792, 7, 7]
    shape_w = [1, 1792, 7, 7]

    op1768 = relay.var("x", shape=shape_x, dtype="float32") #/* ty=Tensor[(1, 1792, 7, 7), float32] */
    op1769 = relay.var("w", shape=shape_w, dtype="float32") #/* ty=Tensor[(1, 1792, 7, 7), float32] */

    #   %1768 = add(%1767, meta_relay_Constant_895] /* ty=Tensor[(1792, 1, 1), float32] */) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1769 = sigmoid(%1768) /* ty=Tensor[(1, 1792, 7, 7), float32] */;

    #   %1770 = multiply(%1768, %1769) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1771 = qnn.quantize(%1770, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
    #   %1772 = qnn.dequantize(%1771, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1773 = nn.global_avg_pool2d(%1772) /* ty=Tensor[(1, 1792, 1, 1), float32] */;
    #   %1774 = squeeze(%1773, axis=[3]) /* ty=Tensor[(1, 1792, 1), float32] */;
    #   %1775 = squeeze(%1774, axis=[2]) /* ty=Tensor[(1, 1792), float32] */;
    #   %1776 = qnn.quantize(%1775, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792), int8] */;
    #   %1777 = qnn.quantize(meta_relay_Constant_896] /* ty=Tensor[(1000, 1792), float32] */, meta_relay_Constant_897] /* ty=Tensor[(1000), float32] */, meta_relay_Constant_898] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
    #   %1778 = qnn.dequantize(%1776, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 1792), float32] */;
    #   %1779 = qnn.dequantize(%1777, meta_relay_Constant_897] /* ty=Tensor[(1000), float32] */, meta_relay_Constant_899] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
    #   %1780 = nn.dense(%1778, %1779, units=1000) /* ty=Tensor[(1, 1000), float32] */;
    #   add(%1780, meta_relay_Constant_900] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */

    meta_relay_Constant_896 = relay.const(np.random.uniform(size=[1000, 1792]).astype("float32")) #/* ty=Tensor[(1000, 1792), float32] */
    meta_relay_Constant_897 = relay.const(np.random.uniform(size=[1000]).astype("float32")) #/* ty=Tensor[(1000), float32] */
    meta_relay_Constant_898 = relay.const(np.random.uniform(size=[1000]).astype("int32")) #/* ty=Tensor[(1000), int32] */
    meta_relay_Constant_899 = relay.const(np.random.uniform(size=[1000]).astype("int32")) #/* ty=Tensor[(1000), int32] */
    meta_relay_Constant_900 = relay.const(np.random.uniform(size=[1000]).astype("float32")) #/* ty=Tensor[(1000), float32] */

    op1770 = relay.op.multiply(op1768, op1769)
    op1771 = relay.qnn.op.quantize(op1770, relay.const(0.179916), relay.const(0), out_dtype="int8", axis=1)
    op1772 = relay.qnn.op.dequantize(op1771, relay.const(0.179916), relay.const(0))
    op1773 = relay.op.nn.global_avg_pool2d(op1772)
    op1774 = relay.op.squeeze(op1773, axis=[3])
    op1775 = relay.op.squeeze(op1774, axis=[2])
    op1776 = relay.qnn.op.quantize(op1775, relay.const(0.0108239), relay.const(0), out_dtype="int8", axis=1)
    op1777 = relay.qnn.op.quantize(meta_relay_Constant_896, meta_relay_Constant_897, meta_relay_Constant_898, out_dtype="int8", axis=0)
    op1778 = relay.qnn.op.dequantize(op1776, relay.const(0.0108239), relay.const(0))
    op1779 = relay.qnn.op.dequantize(op1777, meta_relay_Constant_897, meta_relay_Constant_899, axis=0)
    op1780 = relay.op.nn.dense(op1778, op1779, units=1000)
    expr = relay.op.add(op1780, meta_relay_Constant_900)



    # FakeQuantizationRewriter 
    """
        fn (%x: Tensor[(1, 1792, 7, 7), float32], %w: Tensor[(1, 1792, 7, 7), float32]) -> Tensor[(1, 1000), float32] {
        %0 = multiply(%x, %w) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
        %1 = qnn.quantize(%0, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
        %2 = cast(%1, dtype="int32");
        %3 = nn.global_avg_pool2d(%2);
        %4 = cast(%3, dtype="int8");
        %5 = squeeze(%4, axis=[3]) /* ty=Tensor[(1, 1792, 1), float32] */;
        %6 = squeeze(%5, axis=[2]) /* ty=Tensor[(1, 1792), float32] */;
        %7 = qnn.requantize(%6, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8");
        %8 = qnn.quantize(meta[relay.Constant][0] /* ty=Tensor[(1000, 1792), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][2] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
        %9 = qnn.dequantize(%7, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */) /* ty=Tensor[(1, 1792), float32] */;
        %10 = qnn.dequantize(%8, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][3] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
        %11 = nn.dense(%9, %10, units=1000) /* ty=Tensor[(1, 1000), float32] */;
        add(%11, meta[relay.Constant][4] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */
        }
    """

    # mod tvm.relay.transform.FakeQuantizationToInteger
    """
    def @main(%x: Tensor[(1, 1792, 7, 7), float32], %w: Tensor[(1, 1792, 7, 7), float32]) -> Tensor[(1, 1000), float32] {
    %0 = multiply(%x, %w) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    %1 = qnn.quantize(%0, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
    %2 = cast(%1, dtype="int32") /* ty=Tensor[(1, 1792, 7, 7), int32] */;
    %3 = nn.global_avg_pool2d(%2) /* ty=Tensor[(1, 1792, 1, 1), int32] */;
    %4 = cast(%3, dtype="int8") /* ty=Tensor[(1, 1792, 1, 1), int8] */;
    %5 = squeeze(%4, axis=[3]) /* ty=Tensor[(1, 1792, 1), int8] */;
    %6 = squeeze(%5, axis=[2]) /* ty=Tensor[(1, 1792), int8] */;
    %7 = qnn.requantize(%6, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 1792), int8] */;
    %8 = qnn.quantize(meta[relay.Constant][0] /* ty=Tensor[(1000, 1792), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][2] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
    %9 = qnn.dequantize(%7, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */) /* ty=Tensor[(1, 1792), float32] */;
    %10 = qnn.dequantize(%8, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][3] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
    %11 = nn.dense(%9, %10, units=1000) /* ty=Tensor[(1, 1000), float32] */;
    add(%11, meta[relay.Constant][4] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */
    }
    """
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int32").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int32").astype("float32")

    args = [x_np, w_np]
    allow_rounding_error=False

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False)(mod_def)
    print("mod tvm.relay.transform.InferType\n", mod_def, "\n")
    print("mod tvm.relay.transform.FakeQuantizationToInteger\n", mod_int, "\n")
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    print("result.astype(int32)", result.astype("int32"))
    print("result_int.astype(int32)", result_int.astype("int32"))
    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)



# test_dequantize_propagation()
# test_dequantize_propagation_test()


def test_dequantize_propagation_test2():
    shape_x = [1, 1792, 7, 7]
    shape_w = [1, 1792, 7, 7]

    op1768 = relay.var("x", shape=shape_x, dtype="float32") #/* ty=Tensor[(1, 1792, 7, 7), float32] */
    op1769 = relay.var("w", shape=shape_w, dtype="float32") #/* ty=Tensor[(1, 1792, 7, 7), float32] */


# %bert.encoder.layer.0.attention.self.key.bias: Tensor[(1024), float32]

#   %38 = qnn.quantize(%37, 0.0555795 /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=0) #/* ty=Tensor[(1, 16, 384, 64), int8] */;

#   %39 = qnn.dequantize(%38, 0.0555795 /* ty=float32 */, 0 /* ty=int32 */, axis=0) #/* ty=Tensor[(1, 16, 384, 64), float32] */;
#   %40 = broadcast_to(%39, shape=[1, 16, 384, 64], dtype="")# /* ty=Tensor[(1, 16, 384, 64), float32] */;
#   %41 = qnn.quantize(%25, 0.0196148 /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=0)# /* ty=Tensor[(1, 384, 1024), int8] */;
#   %42 = qnn.quantize(%bert.encoder.layer.0.attention.self.key.weight, 0.00561207 /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=0)# /* ty=Tensor[(1024, 1024), int8] */;
#   %43 = transpose(%42, axes=[1, 0])# /* ty=Tensor[(1024, 1024), int8] */;
#   %44 = reshape(%41, newshape=[-1, 1024]) #/* ty=Tensor[(384, 1024), int8] */;
#   %45 = transpose(%43, axes=None)# /* ty=Tensor[(1024, 1024), int8] */;
#   %46 = qnn.dense(%44, %45, 0 /* ty=int32 */, 0 /* ty=int32 */, 0.0196148 /* ty=float32 */, 0.00561207 /* ty=float32 */, units=None, out_dtype="int32") #/* ty=Tensor[(384, 1024), int32] */;
#   %47 = reshape(%46, newshape=[1, 384, 1024]) #/* ty=Tensor[(1, 384, 1024), int32] */;
#   %48 = qnn.dequantize(%47, 0.00011008 /* ty=float32 */, 0 /* ty=int32 */, axis=1) #/* ty=Tensor[(1, 384, 1024), float32] */;
#   %49 = cast(%bert.encoder.layer.0.attention.self.key.bias, dtype="float32")# /* ty=Tensor[(1024), float32] */;
#   %50 = add(%48, %49) #/* ty=Tensor[(1, 384, 1024), float32] */;
#   %51 = reshape(%50, newshape=[1, 384, 16, 64])# /* ty=Tensor[(1, 384, 16, 64), float32] */;
#   %52 = transpose(%51, axes=[0, 2, 3, 1]) #/* ty=Tensor[(1, 16, 64, 384), float32] */;
#   %53 = qnn.quantize(%52, 0.0555795 /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=0)# /* ty=Tensor[(1, 16, 64, 384), int8] */;
  
#   %54 = qnn.dequantize(%53, 0.0555795 /* ty=float32 */, 0 /* ty=int32 */, axis=0)# /* ty=Tensor[(1, 16, 64, 384), float32] */;
#   %55 = broadcast_to(%54, shape=[1, 16, 64, 384], dtype="")# /* ty=Tensor[(1, 16, 64, 384), float32] */;
#   %56 = reshape(%55, newshape=[-1, 64, 384])# /* ty=Tensor[(16, 64, 384), float32] */;
#   %57 = reshape(%40, newshape=[-1, 384, 64])# /* ty=Tensor[(16, 384, 64), float32] */;
#   %58 = transpose(%56, axes=[0, 2, 1]) #/* ty=Tensor[(16, 384, 64), float32] */;
#   %59 = nn.batch_matmul(%57, %58, out_dtype="float32", transpose_b=True) #/* ty=Tensor[(16, 384, 384), float32] */;


    #   %1768 = add(%1767, meta_relay_Constant_895] /* ty=Tensor[(1792, 1, 1), float32] */) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1769 = sigmoid(%1768) /* ty=Tensor[(1, 1792, 7, 7), float32] */;

    #   %1770 = multiply(%1768, %1769) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1771 = qnn.quantize(%1770, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
    #   %1772 = qnn.dequantize(%1771, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    #   %1773 = nn.global_avg_pool2d(%1772) /* ty=Tensor[(1, 1792, 1, 1), float32] */;
    #   %1774 = squeeze(%1773, axis=[3]) /* ty=Tensor[(1, 1792, 1), float32] */;
    #   %1775 = squeeze(%1774, axis=[2]) /* ty=Tensor[(1, 1792), float32] */;
    #   %1776 = qnn.quantize(%1775, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792), int8] */;
    #   %1777 = qnn.quantize(meta_relay_Constant_896] /* ty=Tensor[(1000, 1792), float32] */, meta_relay_Constant_897] /* ty=Tensor[(1000), float32] */, meta_relay_Constant_898] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
    #   %1778 = qnn.dequantize(%1776, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 1792), float32] */;
    #   %1779 = qnn.dequantize(%1777, meta_relay_Constant_897] /* ty=Tensor[(1000), float32] */, meta_relay_Constant_899] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
    #   %1780 = nn.dense(%1778, %1779, units=1000) /* ty=Tensor[(1, 1000), float32] */;
    #   add(%1780, meta_relay_Constant_900] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */

    meta_relay_Constant_896 = relay.const(np.random.uniform(size=[1000, 1792]).astype("float32")) #/* ty=Tensor[(1000, 1792), float32] */
    meta_relay_Constant_897 = relay.const(np.random.uniform(size=[1000]).astype("float32")) #/* ty=Tensor[(1000), float32] */
    meta_relay_Constant_898 = relay.const(np.random.uniform(size=[1000]).astype("int32")) #/* ty=Tensor[(1000), int32] */
    meta_relay_Constant_899 = relay.const(np.random.uniform(size=[1000]).astype("int32")) #/* ty=Tensor[(1000), int32] */
    meta_relay_Constant_900 = relay.const(np.random.uniform(size=[1000]).astype("float32")) #/* ty=Tensor[(1000), float32] */

    op1770 = relay.op.multiply(op1768, op1769)
    op1771 = relay.qnn.op.quantize(op1770, relay.const(0.179916), relay.const(0), out_dtype="int8", axis=1)
    op1772 = relay.qnn.op.dequantize(op1771, relay.const(0.179916), relay.const(0))
    op1773 = relay.op.nn.global_avg_pool2d(op1772)
    op1774 = relay.op.squeeze(op1773, axis=[3])
    op1775 = relay.op.squeeze(op1774, axis=[2])
    op1776 = relay.qnn.op.quantize(op1775, relay.const(0.0108239), relay.const(0), out_dtype="int8", axis=1)
    op1777 = relay.qnn.op.quantize(meta_relay_Constant_896, meta_relay_Constant_897, meta_relay_Constant_898, out_dtype="int8", axis=0)
    op1778 = relay.qnn.op.dequantize(op1776, relay.const(0.0108239), relay.const(0))
    op1779 = relay.qnn.op.dequantize(op1777, meta_relay_Constant_897, meta_relay_Constant_899, axis=0)
    op1780 = relay.op.nn.dense(op1778, op1779, units=1000)
    expr = relay.op.add(op1780, meta_relay_Constant_900)



    # FakeQuantizationRewriter 
    """
        fn (%x: Tensor[(1, 1792, 7, 7), float32], %w: Tensor[(1, 1792, 7, 7), float32]) -> Tensor[(1, 1000), float32] {
        %0 = multiply(%x, %w) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
        %1 = qnn.quantize(%0, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
        %2 = cast(%1, dtype="int32");
        %3 = nn.global_avg_pool2d(%2);
        %4 = cast(%3, dtype="int8");
        %5 = squeeze(%4, axis=[3]) /* ty=Tensor[(1, 1792, 1), float32] */;
        %6 = squeeze(%5, axis=[2]) /* ty=Tensor[(1, 1792), float32] */;
        %7 = qnn.requantize(%6, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8");
        %8 = qnn.quantize(meta[relay.Constant][0] /* ty=Tensor[(1000, 1792), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][2] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
        %9 = qnn.dequantize(%7, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */) /* ty=Tensor[(1, 1792), float32] */;
        %10 = qnn.dequantize(%8, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][3] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
        %11 = nn.dense(%9, %10, units=1000) /* ty=Tensor[(1, 1000), float32] */;
        add(%11, meta[relay.Constant][4] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */
        }
    """

    # mod tvm.relay.transform.FakeQuantizationToInteger
    """
    def @main(%x: Tensor[(1, 1792, 7, 7), float32], %w: Tensor[(1, 1792, 7, 7), float32]) -> Tensor[(1, 1000), float32] {
    %0 = multiply(%x, %w) /* ty=Tensor[(1, 1792, 7, 7), float32] */;
    %1 = qnn.quantize(%0, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 1792, 7, 7), int8] */;
    %2 = cast(%1, dtype="int32") /* ty=Tensor[(1, 1792, 7, 7), int32] */;
    %3 = nn.global_avg_pool2d(%2) /* ty=Tensor[(1, 1792, 1, 1), int32] */;
    %4 = cast(%3, dtype="int8") /* ty=Tensor[(1, 1792, 1, 1), int8] */;
    %5 = squeeze(%4, axis=[3]) /* ty=Tensor[(1, 1792, 1), int8] */;
    %6 = squeeze(%5, axis=[2]) /* ty=Tensor[(1, 1792), int8] */;
    %7 = qnn.requantize(%6, 0.179916f /* ty=float32 */, 0 /* ty=int32 */, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 1792), int8] */;
    %8 = qnn.quantize(meta[relay.Constant][0] /* ty=Tensor[(1000, 1792), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][2] /* ty=Tensor[(1000), int32] */, out_dtype="int8", axis=0) /* ty=Tensor[(1000, 1792), int8] */;
    %9 = qnn.dequantize(%7, 0.0108239f /* ty=float32 */, 0 /* ty=int32 */) /* ty=Tensor[(1, 1792), float32] */;
    %10 = qnn.dequantize(%8, meta[relay.Constant][1] /* ty=Tensor[(1000), float32] */, meta[relay.Constant][3] /* ty=Tensor[(1000), int32] */, axis=0) /* ty=Tensor[(1000, 1792), float32] */;
    %11 = nn.dense(%9, %10, units=1000) /* ty=Tensor[(1, 1000), float32] */;
    add(%11, meta[relay.Constant][4] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */
    }
    """
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int32").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int32").astype("float32")

    args = [x_np, w_np]
    allow_rounding_error=False

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False)(mod_def)
    print("mod tvm.relay.transform.InferType\n", mod_def, "\n")
    print("mod tvm.relay.transform.FakeQuantizationToInteger\n", mod_int, "\n")
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    print("result.astype(int32)", result.astype("int32"))
    print("result_int.astype(int32)", result_int.astype("int32"))
    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)




# test_dequantize_propagation_test2()

def test_dequantize_propagation_broadcast_to():
    shape_x = [1, 4, 2]
    # shape_w = [8]
    shape_w = [1]
    # shape_w = [1, 4, 2]


    # test_broadcast_to((1, 1, 5, 4), (3, 4, 4, 4, 5, 4))
    # shape_w = [1, 4, 2]
    # shape_w = [1, 8, 2]
    x = relay.var("x", shape=shape_x, dtype="int8")
    w = relay.var("w", shape=shape_w, dtype="int8")

    # a = relay.qnn.op.dequantize(x, relay.const(1.5), relay.const(0)) # input, scale, shift
    # b = relay.qnn.op.dequantize(w, relay.const(0.5), relay.const(0)) # input, scale, shift
    # op = relay.op.nn.batch_matmul(a, b)
    # op = relay.op.add(op, relay.const(2.0, "float32"))
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type


    # a = relay.qnn.op.dequantize(x, relay.const(1.5), relay.const(0)) # input, scale, shift
    # b = relay.qnn.op.dequantize(w, relay.const(0.5), relay.const(0)) # input, scale, shift
    # op = relay.op.nn.batch_matmul(a, b)
    # op = relay.op.add(a, b)
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.op.subtract(op, relay.const(1, "int8"))

    # op = relay.op.erf(op) # HERE
    # op = relay.op.multiply(op, relay.const(3.0))
    # op = relay.op.subtract(op, relay.const(1, "int8"))
    
    a = x
    b = w

    # a = relay.op.abs(x)
    # b = relay.op.abs(w)

    # a = relay.op.add(a, relay.const(2.0, "int8"))
    # b = relay.op.add(b, relay.const(3.0, "int8"))

    a = relay.qnn.op.dequantize(a, relay.const(1.5), relay.const(0)) # input, scale, shift
    b = relay.qnn.op.dequantize(b, relay.const(0.5), relay.const(0)) # input, scale, shift

    # op = relay.op.add(a, b)

    # op1 = relay.op.nn.batch_matmul(a, b)
    # op2 = relay.op.nn.batch_matmul(b, a)
    # op11 = relay.op.nn.batch_matmul(op1, op2)
    # op22 = relay.op.nn.batch_matmul(op2, op1)
    # op = relay.op.nn.batch_matmul(op11, op22)

        # broadcast_to

    ax = relay.op.broadcast_to(b, (3,3))
    # ax = a
    op = ax
    # op = relay.op.nn.batch_matmul(ax, b)
    # op2 = relay.op.nn.batch_matmul(b, a)
    # op = relay.op.nn.batch_matmul(op, op2)
    # op = relay.op.add(op, relay.const(2.0, "float32"))

    # op = relay.op.erf(op) # here
    # op = relay.op.multiply(op, relay.const(3.0))
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.qnn.op.quantize(op, relay.const(2.5), relay.const(0), out_dtype="int8") #input, scale, shift, type
    # op = relay.op.subtract(op, relay.const(1, "int8"))

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")

    expr = op
    args = [w_np]
    # args = [x_np, w_np]
    allow_rounding_error=False

    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False)(mod_def)
    print("mod tvm.relay.transform.InferType\n", mod_def, "\n")
    print("mod tvm.relay.transform.FakeQuantizationToInteger\n", mod_int, "\n")
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    print("result.astype(int32)", result.astype("int32"))
    print("result_int.astype(int32)", result_int.astype("int32"))
    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)



test_dequantize_propagation_broadcast_to()