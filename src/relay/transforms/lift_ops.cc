/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file canonicalize_ops.cc
 * \brief Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class RequantizeSimplifier : public ExprRewriter {
 public:
  RequantizeSimplifier() : rq_op_(Op::Get("qnn.requantize")),
                           transpose_op_(Op::Get("transpose")),
                           reshape_op_(Op::Get("reshape")) {}

  /*Expr Rewrite_(const CallNode* n, const Expr& post) override {
    if (n->op == rq_op_) {
      const auto pred = n->args[0].as<CallNode>();
      if (pred && pred->op == transpose_op_) {
        const auto rq_pred = post.as<CallNode>()->args[0];
        const auto transpose_pred = rq_pred.as<CallNode>()->args[0];

        // Create new qnn.requantize
        tvm::Array<relay::Expr> rq_args_new = n->args;
        rq_args_new.Set(0, transpose_pred);
        auto rq_new = Call(rq_op_, rq_args_new, n->attrs, n->type_args);

        return Call(transpose_op_, {rq_new}, pred->attrs, pred->type_args);
      }
    }
    return post;
  }*/

  Expr Rewrite_(const CallNode* n, const Expr& post) override {
    // Return if operation is not qnn.requantize.
    if (n->op != rq_op_)
      return post;

    std::stack<const CallNode*> passed;
    const Expr& pred = post.as<CallNode>()->args[0];
    const Expr& term = FindTermExpr(pred, passed);
    if (term == pred)
      return post;

    // Create new qnn.requantize
    tvm::Array<relay::Expr> rq_args_new = n->args;
    rq_args_new.Set(0, term);
    auto rq_new = Call(rq_op_, rq_args_new, n->attrs, n->type_args);

    auto expr_pred = rq_new;
    while (!passed.empty()) {
      const CallNode* passed_node = passed.top();
      expr_pred = Call(passed_node->op, {expr_pred},
                       passed_node->attrs, passed_node->type_args);
      passed.pop();
    }
    return expr_pred;
  }

 private:
  // Iterate through the predecessor and find first not supported
  // expression and return this node (term node).
  // Supported ops - reshape and transpose.
  const Expr& FindTermExpr(const Expr& e, std::stack<const CallNode*>& passed) {
    const auto cn = e.as<CallNode>();
    if (cn && (cn->op == transpose_op_ || cn->op == reshape_op_)) {
      const Expr& pred = cn->args[0];
      passed.push(cn);
      return FindTermExpr(pred, passed);
    } else {
      return e;
    }
  }

  const Op& rq_op_;
  const Op& transpose_op_;
  const Op& reshape_op_;
};

Expr LiftOps(const Expr& e) {
  auto rewriter = RequantizeSimplifier();
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass LiftOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(LiftOps(f));
      };
  return CreateFunctionPass(pass_func, 0, "LiftOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.LiftOps").set_body_typed(LiftOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
