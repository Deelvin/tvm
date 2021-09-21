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
        self.input_scale = None
        self.input_zero_point = None
        self.processed = 0

        self.in_qnn_add = False
        self.branches = {}
        super().__init__()

    def visit_call(self, call):
        # Visit current call operation
        new_fn = self.visit(call.op)
        # Visit current arguments
        if call.op == relay.op.get("qnn.add"):
          # saving data for adding dequantize just before previous op
          print("inadd!!!")
          self.in_qnn_add = call
          self.branches[call] = False
          a0 = self.visit(call.args[0])
          if self.branches[call] is False:
            a0 = relay.qnn.op.dequantize(a0, call.args[2], call.args[3])
          self.in_qnn_add = call
          self.branches[call] = False
          a1 = self.visit(call.args[1])
          if self.branches[call] is False:
            a1 = relay.qnn.op.dequantize(a1, call.args[4], call.args[5])
          op_add = relay.add(a0, a1)
          self.in_qnn_add = None
          self.branches[call] = False
          return relay.qnn.op.quantize(op_add, self.visit(call.args[6]), self.visit(call.args[7]), out_dtype="uint8")

        if call.op == relay.op.get("qnn.quantize") and self.in_qnn_add is not None:
          self.branches[self.in_qnn_add] = True
          self.in_qnn_add = None
          
          #return self.visit(relay.cast(call.args[0], , )
          print("in quantize!!!")
          return self.visit(call.args[0])

        if call.op == relay.op.get("cast") and self.in_qnn_add is not None:
          #self.in_qnn_add = False
          #return self.visit(relay.cast(call.args[0], , )
          print("incast!!!!")
          return self.visit(call.args[0])

        if call.op == relay.op.get("clip") and self.in_qnn_add is not None:
          #self.in_qnn_add = False
          print("inclip!!!!")
          #return self.visit(relay.cast(call.args[0], , )
          return self.visit(call.args[0])

        # Otherwise return the unchanged call.
        # Visit current arguments
        print(call.op)
        self.in_qnn_add = None
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return Call(new_fn, args, call.attrs)

def dequantize_qadd(expr):
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
