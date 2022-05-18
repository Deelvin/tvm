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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging

import tvm.ir
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import GlobalVar
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import const

from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table


def get_dnnl_version():
    """Return tuple with version or DNNL library if known
    Otherwise return unknown value which is bigger than any over real
    versions.
    """
    f = tvm.get_global_func("runtime.module.dnnl_version", allow_missing=True)
    return tuple(int(el) for el in f().split(".")) if f else (100500,)


dnnl_version = get_dnnl_version()

logger = logging.getLogger("DNNL")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        args = expr.args
        if any([x.checked_type.dtype == "int64" for x in args]):
            logger.info("DNNL does not support int64.")
            return False
        # Binary ops with asymmetrical shapes of arguments (need broadcasting
        # for one of the arguments) use reference implementration of primitive
        # and have poor performance. So, let's skip such ops.
        if op_name in ("add", "multiply"):
            if list(args[0].checked_type.shape) != list(args[1].checked_type.shape):
                return False
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    """Create patterns related to conv and conv_transpose.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `conv / conv_transpose`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


def make_dnnl_pattern(op_name, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op_name : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = op_name.replace("nn", "dnnl")
    if "_transpose" in op_name:
        pat_name = "dnnl.deconv" + op_name.split("_")[0][-2::]
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if "conv" in op_name:
        dnnl_pattern = (pat_name, make_conv_pattern(op_name, with_bias, with_eltwise))
    elif op_name == "nn.dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning(
            "Currently, only conv1d, conv2d, conv2d_transpose, conv3d_transpose and "
            "dense op are supported, but got %s.",
            op_name,
        )
        dnnl_pattern = ()
    return dnnl_pattern


def make_qnn_conv2d_pattern(with_sum=False):
    """Make qnn.conv2d based pattern supported by DNNL

    Parameters
    ----------
    with_sum : bool
        Indicate to append qnn.sum at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    weight = is_constant()  # |const requirements, have to recalculate bias to compensate src_zp
    bias = is_constant()

    pat = wildcard()
    pat = is_op("qnn.conv2d")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("add")(pat, bias) | pat
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(
            pat,
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pat = is_op("clip")(pat)

    pat_name = "dnnl.qnn.conv2d_sum" if with_sum else "dnnl.qnn.conv2d"

    return pat_name, pat


def make_qnn_dense_pattern(with_sum=False):
    """Make qnn.dense based pattern supported by DNNL

    Parameters
    ----------
    with_sum : bool
        Indicate to append qnn.sum at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    weight = is_constant()
    bias = is_constant()

    pat = wildcard()
    pat = is_op("qnn.dense")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("add")(pat, bias) | pat
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(
            pat,
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pat = is_op("clip")(pat)

    pat_name = "dnnl.qnn.dense_sum" if with_sum else "dnnl.qnn.dense"

    return pat_name, pat


def make_pattern_qnn_dense_dequantize_bias_gelu(with_gelu=False):
    """Make qnn.dense based pattern supported by DNNL

    Parameters
    ----------
    with_gelu : bool
        Indicate to append Gelu pattern at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat = wildcard()
    weight = wildcard()
    pat = is_op("qnn.dense")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, is_constant(), is_constant())
    pat = is_op("add")(pat, is_constant())
    if with_gelu is True:
        div = is_op("divide")(pat, is_constant())
        erf = is_op("erf")(div)
        mul = is_op("multiply")(pat, is_expr(const(0.5, dtype="float32")))
        pat = is_op("add")(erf, is_expr(const(1.0, dtype="float32")))
        pat = is_op("multiply")(mul, pat)
    pat = is_op("qnn.quantize")(pat, is_constant(), is_constant())
    pat_name = "dnnl.qnn.dense_dequantize_bias_gelu"
    return pat_name, pat


def make_pattern_qnn_dense_bias_requantize():
    """Make qnn.dense based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat = wildcard()
    weight = wildcard()
    bias = is_constant()
    pat = is_op("qnn.dense")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.add")(
        pat,
        bias,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pat = is_op("reshape")(pat) | pat
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat_name = "dnnl.qnn.dense_bias_requantize"
    return pat_name, pat


def make_pattern_qnn_matmul_requantize():
    """Make qnn.batch_matmul based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat = wildcard()
    weight = is_op("transpose")(wildcard())
    pat = is_op("qnn.batch_matmul")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat_name = "dnnl.qnn.matmul_requantize"
    return pat_name, pat


def make_pattern_qnn_transpose_matmul_dequantize(with_div):
    """Make qnn.batch_matmul based pattern supported by DNNL

    Parameters
    ----------
    with_div : bool
        Indicate to append nn.div at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat = wildcard()
    weight = is_op("transpose")(wildcard())
    pat = is_op("qnn.batch_matmul")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, is_constant(), is_constant())
    if with_div is True:
        pat = is_op("divide")(pat, is_constant())

    pat_name = "dnnl.qnn.matmul_dequantize_div" if with_div else "dnnl.qnn.matmul_dequantize"

    return pat_name, pat


def make_pattern_normalization():
    """Make layer normalization based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat = wildcard()
    mean = is_op("mean")(pat)
    subt = is_op("subtract")(pat, mean)
    power = is_op("power")(subt, is_expr(const(2, dtype="float32")))
    pat = is_op("mean")(power)
    pat = is_op("add")(pat, is_constant())
    pat = is_op("sqrt")(pat)
    pat = is_op("divide")(subt, pat)
    pat = is_op("multiply")(pat, is_constant())
    pat = is_op("add")(pat, is_constant())
    pat_name = "dnnl.layer.normalize"
    return pat_name, pat


def make_pattern_softmax_quantize():
    pat = is_op("nn.softmax")(wildcard())
    pat = is_op("qnn.quantize")(pat, is_constant(), is_constant())
    pat_name = "dnnl.softmax_quantize"
    return pat_name, pat


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    dnnl_patterns = []

    for with_sum in [True, False]:
        dnnl_patterns.append(make_qnn_conv2d_pattern(with_sum))
        # Old dnnl version doesn't support per channel o_scale
        if dnnl_version >= (2, 2) or not with_sum:
            dnnl_patterns.append(make_qnn_dense_pattern(with_sum))

    dnnl_patterns.append(make_pattern_qnn_dense_dequantize_bias_gelu(True))
    dnnl_patterns.append(make_pattern_qnn_dense_bias_requantize())
    dnnl_patterns.append(make_pattern_qnn_matmul_requantize())
    for with_div in [True, False]:
        dnnl_patterns.append(make_pattern_qnn_transpose_matmul_dequantize(with_div))
    dnnl_patterns.append(make_pattern_normalization())
    if dnnl_version >= (2, 6):
        dnnl_patterns.append(make_pattern_softmax_quantize())

    elt_list = ["nn.relu", "tanh", "sigmoid", None]
    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                return dnnl_patterns
            for conv_name in [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv3d",
                "nn.conv2d_transpose",
                "nn.conv3d_transpose",
            ]:
                dnnl_patterns.append(make_dnnl_pattern(conv_name, with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("nn.dense", with_bias, elt))

    return dnnl_patterns


def get_optimal_layout_for_conv(
    data_layout, kernel_layout, weight_shape, out_shape, paddings, strides, dilates, groups
):
    """Get the optimal layout of dnnl, given shape of conv2d.

    Parameters
    ----------
    data_layout, kernel_layout,weight_shape, out_shape, paddings, strides, dilates, groups
        : String
          Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv(
        data_layout,
        kernel_layout,
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        groups,
    )


def get_optimal_layout_for_conv_transpose(
    data_layout,
    kernel_layout,
    weight_shape,
    out_shape,
    paddings,
    output_paddings,
    strides,
    dilates,
    groups,
):
    """Get the optimal layout of dnnl, given shape of tranposed conv2d.

    Parameters
    ----------
    data_layout, kernel_layout, weight_shape, out_shape, paddings, output_paddings, strides,
    dilates, groups
        : Int, String
          Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv_transpose(
        data_layout,
        kernel_layout,
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
    )


def get_shape(tensor):
    """Get tensor's shape."""
    if isinstance(tensor, relay.expr.Var):
        return tensor.type_annotation.concrete_shape
    if isinstance(tensor, relay.expr.Constant):
        return tensor.data.shape
    if isinstance(tensor, tvm.ir.tensor_type.TensorType):
        return tensor.concrete_shape
    if isinstance(tensor, tvm.ir.container.Array):
        return tensor[-1].shape
    if isinstance(tensor, relay.expr.Call):
        return tensor.checked_type.shape
    raise TypeError("Unsupport data type: %s" % type(tensor))


def tag2layout(input_data, is_weight=False, conv_type="Conv1D"):
    """Transfer layout, denoted with `a, b, c, d, e`,
    into valid layout (NCHW / OIHW) of TVM."""
    if "Conv1D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "W", "d": "G"}
    elif "Conv2D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "H", "d": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "H", "d": "W"}
        if "e" in input_data:
            weight_dic = {"a": "G", "b": "O", "c": "I", "d": "H", "e": "W"}
    elif "Conv3D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "D", "d": "H", "e": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "D", "d": "H", "e": "W", "f": "G"}

    dic = weight_dic if is_weight else data_dic
    res = ""

    for i in input_data:
        if i.isupper():
            i = i.lower()
            res += dic[i]
            dic[i] = dic[i].lower()
        elif i.islower():
            res += dic[i]
        elif i.isdigit():
            res += i
        else:
            raise ValueError("Unsupport layout format: %s" % input_data)
    return res


def legalize_group_conv(attrs, inputs, types):
    """Legalize group conv / conv_transpose calculation.
    Alter weight layout from OIHW to GOIHW / IOHW to GIOHW"""
    groups = attrs.groups
    data, weight = inputs
    if groups == 1:
        if "Transpose" not in type(attrs).__name__:
            return relay.nn.conv2d(data, weight, **attrs)
        return relay.nn.conv2d_transpose(data, weight, **attrs)
    OC, IC, H, W = get_shape(weight)
    new_attrs = dict(attrs)
    weight = relay.reshape(weight, (groups, OC // groups, IC, H, W))
    if "Transpose" not in type(attrs).__name__:
        new_attrs["kernel_layout"] = "GOIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)
    new_attrs["kernel_layout"] = "GIOHW"
    return relay.nn.conv2d_transpose(data, weight, **new_attrs)


def alter_conv(attrs, inputs, tinfos, out_type):
    """The convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    groups = str(attrs.groups)
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv(
        attrs["data_layout"],
        attrs["kernel_layout"],
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        groups,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1D":
        return relay.nn.conv1d(data, weight, **new_attrs)
    if conv_type == "Conv2D":
        return relay.nn.conv2d(data, weight, **new_attrs)
    return relay.nn.conv3d(data, weight, **new_attrs)


def alter_conv_transpose(attrs, inputs, tinfos, out_type):
    """The transposed convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    output_paddings = ",".join([str(x) for x in attrs.get_int_tuple("output_padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    groups = str(attrs.groups)
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv_transpose(
        attrs["data_layout"],
        attrs["kernel_layout"],
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1DTranspose":
        return relay.nn.conv1d_transpose(data, weight, **new_attrs)
    if conv_type == "Conv2DTranspose":
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)
    return relay.nn.conv3d_transpose(data, weight, **new_attrs)


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains compute heavy ops like convolutions and
    its transpose and dense.
    """

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        compute_intensive_ops = set(
            [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv2d_transpose",
                "nn.conv3d",
                "nn.conv3d_transpose",
                "nn.dense",
            ]
        )
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op) in compute_intensive_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph) -> bool:
        """
        This function recursively visits the graph and checks if it's compute intensive"
        """
        self.visit(subgraph)
        return self.is_compute_intensive


def is_valid_subgraph(body):
    """Final check on whether the subgraph is valid and should be offloaded to DNNL."""
    return IsComputeIntensiveGraph().is_graph_compute_intensive(body)


def prune_dnnl_subgraphs(mod):
    """
    Removes invalid subgraphs, which does not contain compute intensive dnnl ops.
    """

    class SubgraphRemover(ExprMutator):
        """
        Reverts subgraphs in subgraphs_to_remove back to TVM instead of using an external codegen.
        """

        def __init__(self, subgraphs_to_remove, mod, new_mod):
            ExprMutator.__init__(self)
            self.subgraphs_to_remove = subgraphs_to_remove
            self.mod = mod
            self.new_mod = new_mod

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if name in self.subgraphs_to_remove:
                    # "Inline" the subgraph back into new main function.
                    func = self.mod[name]
                    var_map = {}
                    for arg, param in zip(call.args, func.params):
                        var_map[param] = super().visit(arg)
                    new_body = relay.bind(func.body, var_map)
                    return new_body
                if name != "main":
                    args = []
                    for arg in call.args:
                        args.append(super().visit(arg))
                    return call.op(*args)
            return super().visit_call(call)

    subgraphs_to_remove = []
    # If only one subgraph, do nothing.
    if len(mod.get_global_vars()) <= 2:
        return mod
    # Remove invalid subgraphs
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != "dnnl":
            continue
        if not is_valid_subgraph(mod[name].body):
            subgraphs_to_remove.append(name)
    # Create new pruned module
    new_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    new_mod["main"] = SubgraphRemover(subgraphs_to_remove, mod, new_mod).visit(mod["main"])
    new_mod = transform.RemoveUnusedFunctions()(new_mod)
    return new_mod
