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
"""Relay type recasting pass"""
import tvm
from tvm import relay
from tvm.ir import IRModule
from .transform import InferType
from ..analysis import count_layers
from ..expr_functor import ExprMutator, Call


class RecastMutator(ExprMutator):
    """Cast operations to the target type."""

    def __init__(self):
        self.before_recast = False
        self.input_scale = None
        self.input_zero_point = None
        self.processed = 0
        super().__init__()

    def visit_call(self, call):
        # Visit current call operation
        new_fn = self.visit(call.op)
        # Visit current arguments
        if call.op == relay.op.get("qnn.requantize"):
          # saving data for adding dequantize just before previous op
          self.before_recast = True
          args = []
          for arg in call.args:
              args.append(self.visit(arg))
          return Call(new_fn, args, call.attrs)
        elif self.before_recast == True:
          self.before_recast = False
          args = [self.visit(arg) for arg in call.args]
          new_args = list()
          for arg in args:
              new_args.append(relay.cast(arg, "float32"))
          return Call(new_fn, new_args, call.attrs)

        # Otherwise return the unchanged call.
        # Visit current arguments
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return Call(new_fn, args, call.attrs)

def cast_postops_float32(expr):
    return_mod = False
    if isinstance(expr, tvm.ir.IRModule):
        expr = expr["main"]
        return_mod = True
    recast_pass = RecastMutator()
    expr = recast_pass.visit(expr)
    print("finished dequantize_postops")
    if return_mod:
        return tvm.IRModule.from_expr(expr)
    return expr
