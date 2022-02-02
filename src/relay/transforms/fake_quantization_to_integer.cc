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
 * \file src/relay/transforms/quantize_fake_quantization.cc
 * \brief A pass for taking fake quantized graphs and converting them
 * to actual integer operations.
 */

#include <tvm/ir/affine_type.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

#include "../src/relay/qnn/op/dequantize.h"
#include "../src/relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using ExprMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
using AffineTypeMap = Map<Expr, AffineType>;

using FTVMFakeQuantizationToInteger =
    runtime::TypedPackedFunc<Array<ObjectRef>(const Expr& expr, const AffineTypeMap& map)>;

template <typename T>
std::string to_expr_str(const T& expr) {
  if (const CallNode* call_node = expr.as<CallNode>()) {
    std::ostringstream os;
    os << call_node->op << " " << call_node;
    return os.str();
  }
  return "<null>";
}

bool is_enabled_op(const CallNode* call_node) {
  const Op op = Downcast<Op>(call_node->op);
  static auto fqfq = Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");

  static std::set<Op> ops = {
    // Op::Get("qnn.dequantize"),
    // Op::Get("qnn.quantize"),
    // Op::Get("qnn.requantize"),

    // register_unary_identity
    Op::Get("reshape"),
    Op::Get("squeeze"),
    Op::Get("strided_slice"),
    Op::Get("transpose"),
    Op::Get("expand_dims"),
    Op::Get("nn.max_pool2d"),
    Op::Get("nn.batch_flatten"),
    Op::Get("nn.depth_to_space"),
    Op::Get("max"),
    Op::Get("min"),

    Op::Get("nn.avg_pool2d"),
    Op::Get("nn.global_avg_pool2d"),
    Op::Get("nn.bias_add"),
    // Op::Get("rsqrt"), // output_scale
    Op::Get("nn.conv2d"),
    Op::Get("nn.conv2d_transpose"),
    Op::Get("nn.dense"),
    Op::Get("nn.batch_matmul"),
    // Op::Get("concatenate"), // output_scale
    Op::Get("split"),
    Op::Get("clip"),
    Op::Get("nn.relu"),
    Op::Get("nn.pad"),

    // register_binary_qnn
    // Op::Get("add"), // output_scale
    // Op::Get("multiply"), // output_scale
    // Op::Get("subtract"), // output_scale

    // register_binary_identity
    Op::Get("minimum"),
    Op::Get("maximum")
  };

  auto is_enabled = [&](const auto i) { return i == call_node->op; };

  auto result = std::find_if(std::begin(ops), std::end(ops), is_enabled);

  return result != ops.end() && fqfq.count(Downcast<Op>(op));
  // return call_node->op != dequantize_op_ && call_node->op != quantize_op_ &&  fqfq.count(Downcast<Op>(op));
}

class SubgraphExtractorOne : public ExprVisitor {
 public:
  const CallNode* expr_call_node;
  const ExprSet GetSubgraph(const Expr& expr) {
    std::string expr_str = to_expr_str(expr);
    // // std::cout << ">>>>>> seo::getsubgraph " << expr << std::endl;
    // std::cout << ">>>>>> seo::getsubgraph " << expr_str << std::endl;
    /* const CallNode* */ expr_call_node = expr.as<CallNode>();
    ICHECK(expr_call_node != nullptr);
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    ICHECK(is_enabled_op(expr_call_node));
    //ICHECK(fqfq.count(Downcast<Op>(expr_call_node->op)));

    VisitExpr(expr);

    ExprSet subgraph;
    // // std::cout << "collect subgraph" << std::endl;
    if (is_fake_quantized_) {  //?
      for (auto kv : this->visit_counter_) {
        if (auto call_node = GetRef<ObjectRef>(kv.first).as<CallNode>()) {
          // // std::cout << "k " << call_node->op << " " << call_node << std::endl;

          if (call_node != expr_call_node) {  //?
            // // std::cout << "v " << kv.second << std::endl;
            subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
          }
        }
      }
    }
    // // std::cout << "/collect subgraph" << std::endl;
    return subgraph;
  }
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
    // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
    // abort the rewrite.
    std::string expr_str = to_expr_str(expr);
    // std::cout << ">>>>>> SEO::VisitExpr " << expr_str << std::endl;
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      // // std::cout << "FakeQuantizationToInteger found a non - dataflow op inside a fake quantize region, aborting this rewrite" << std::endl;
      DLOG(INFO) << "FakeQuantizationToInteger found a non - dataflow op inside a fake quantize "
                    "region, aborting this rewrite";
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    // std::cout << ">>>>>> SEO::VisitExpr_1 " << call_node->op << " " << call_node << std::endl;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");

    if (call_node->op == quantize_op_) {
      // pass
    } else if (call_node->op == dequantize_op_) {
      const auto* attrs = call_node->attrs.as<qnn::DequantizeAttrs>();
      ICHECK(attrs != nullptr);
      // Collect type of dequantize ops
      // // std::cout << "SEO::VisitExpr_1 affine_types_.Set " << GetRef<Expr>(call_node) << std::endl;

      auto e = GetRef<Expr>(call_node);
      auto a1 = call_node->args[1];
      auto a2 = call_node->args[2];
      auto a0 = tvm::relay::transform::InferTypeLocal(call_node->args[0]).as<TensorTypeNode>()->dtype;
      //auto a0 = call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype;
      auto axis = attrs->axis;
      //auto axis = attrs->axis;
      auto t = TensorAffineType(a1, a2, a0, axis);
      affine_types_.Set(e, t);
      //affine_types_.Set(
      //    GetRef<Expr>(call_node),
      //    TensorAffineType(call_node->args[1], call_node->args[2],
      //                     call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype,
      //                     attrs->axis));
    } else if (call_node == expr_call_node) {
    //} else if (fqfq.count(Downcast<Op>(call_node->op))) {
      // } else if (call_node->op == batch_matmul_op_) {
      // const auto* attrs = call_node->attrs.as<BatchMatmulAttrs>();  // ICE
      // ICHECK(attrs != nullptr);

      // VisitExpr(call_node->args[0]);// for abs
      // VisitExpr(call_node->args[1]);// for abs
      // // std::cout << "call_node->args.size()" << call_node->args.size() << std::endl;
      for (auto arg : call_node->args) {
        // for (auto arg : call_node->args) {
        // ExprVisitor::VisitExpr(arg);
        VisitExpr(arg);
      }
      // Collect type of dequantize ops
      // n->scale = std::move(scale);
      // n->zero_point = std::move(zero_point);
      // n->dtype = std::move(dtype);
      // n->axis = std::move(axis);
      // data_ = std::move(n);

      

      auto t = TensorAffineType(MakeConstantScalar(DataType::Int(32), 88),
                                MakeConstantScalar(DataType::Int(32), 44),
                                tvm::relay::transform::InferTypeLocal(call_node->args[0]).as<TensorTypeNode>()->dtype, 1);
                                //call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype, 1);
      // // std::cout << "SEO::VisitExpr_1 affine_types_.Set " << GetRef<Expr>(call_node) << std::endl;
      affine_types_.Set(GetRef<Expr>(call_node), t);
    } else {
      // run normally on everything else.
      ExprVisitor::VisitExpr_(call_node);
    }
  }

  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
};

class SubgraphMutatorOne : public ExprMutator {
 public:
  SubgraphMutatorOne(ExprSet subgraph, AffineTypeMap affine_types, bool hard_fail)
      : subgraph_(subgraph), affine_types_(affine_types), hard_fail_(hard_fail) {}

  Expr MutateSubgraph(const Expr& expr) {
    std::string expr_str = to_expr_str(expr);
    // std::cout << ">>>>>> SMO::MutateSubgraph " << expr_str << std::endl;
    if (subgraph_.size() == 0) {
      return expr;
    }

    // const CallNode*
    quantize_node = expr.as<CallNode>();
    ICHECK(quantize_node);
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    ICHECK(is_enabled_op(quantize_node));
    //ICHECK(fqfq.count(Downcast<Op>(quantize_node->op)));
    out_type_ = affine_types_[expr];  ///???

    // // std::cout << "affine_types_" << std::endl;
    // for (const auto& it : affine_types_) {
    //   // // std::cout << "k " << it.first << std::endl;
    //   // // std::cout << "v " << it.second << std::endl;
    // }
    // // std::cout << "/affine_types_" << std::endl;
    // std::cout << "subgraph_" << std::endl;

    for (auto node : subgraph_) {
      const Op op = Downcast<Op>(node.as<CallNode>()->op);
      std::string expr_str = to_expr_str(node);
      // // std::cout << "node " << expr_str << std::endl;
      //if (!is_enabled_op(node.as<CallNode>())) {
      if (!fqfq.count(Downcast<Op>(op))) {
        // Only modify the subgraph if we have translation
        // rules for every op
        if (hard_fail_) {
          LOG(FATAL) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
        } else {
          DLOG(INFO) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
          return expr;
        }
      }
    }
    // std::cout << "/subgraph_" << std::endl;
    try {
      return Mutate(expr);
    } catch (std::exception& e) {
      if (hard_fail_) {
        throw e;
      } else {
        // // std::cout << "Ran into an error rewriting a subgraph, skipping " << expr << std::endl;
        DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping" << expr << std::endl;
        return expr;
      }
    }
  }

 protected:
  const CallNode* quantize_node;
  Expr VisitExpr_(const CallNode* call_node) {
    // std::cout << ">>>>>> SMO::VisitExpr_ " << call_node->op << " " << call_node << std::endl;
    Expr out;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");

    Op op = Downcast<Op>(call_node->op);
    //if (is_enabled_op(call_node)) {
    if (fqfq.count(op)) {
      Expr expr;
      if (op == dequantize_op_) {
        // // // std::cout << "dequantize_op_2 op->op_type->arg_types.size() " <<
        // op->op_type->arg_types.size() << std::endl; // // std::cout << "dequantize_op_ op->arguments "
        // << std::endl;

        // // // std::cout << "dequantize_op_ op->op_type " << op->op_type->arg_types << std::endl;
        // // // std::cout << "dequantize_op_ op->description " << op->description << std::endl;

        expr = GetRef<Expr>(call_node);
      } else {
        expr = ExprMutator::VisitExpr_(call_node);
        // Set the current op to the output type, useful if we can't deduce output parameters
        // from input parameters
        // // std::cout << "SMO::VisitExpr_ affine_types_.Set " << expr << std::endl;
        affine_types_.Set(expr, out_type_);  // ???
      }
      // Call the rewrite
      Array<ObjectRef> vals = fqfq[op](expr, affine_types_);  // [out, t]
      // Save the outputs of the rewrite
       // // std::cout << "vals size " << vals.size() << std::endl;
      ICHECK(vals.size() == 2)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);

      // // std::cout << "SMO::VisitExpr_ affine_types_.Set " << out << std::endl;
      affine_types_.Set(out, Downcast<AffineType>(vals[1]));
       // // std::cout << "common out "<< out << std::endl;

      const AffineType& tt = Downcast<AffineType>(vals[1]);
      if (call_node == quantize_node) {
      //if (op != dequantize_op_) {
        // // std::cout << "add dequantize_op_ after " << op << " " << call_node << std::endl;

        try {
          // // std::cout << "data " << to_expr_str(out) << std::endl;
          // // std::cout << "scale " << vals[1].as<TensorAffineTypeNode>()->scale << std::endl;
          // // std::cout << "zero_point " << vals[1].as<TensorAffineTypeNode>()->zero_point << std::endl;
          // // std::cout << "axis " << vals[1].as<TensorAffineTypeNode>()->axis << std::endl;

          out = qnn::MakeDequantize(out, vals[1].as<TensorAffineTypeNode>()->scale,
                                    vals[1].as<TensorAffineTypeNode>()->zero_point,
                                    vals[1].as<TensorAffineTypeNode>()->axis);
           //// // std::cout << "nodeq out " << out-> << std::endl;
          // // std::cout << "SMO::VisitExpr_ affine_types_.Set " << out << std::endl;
          affine_types_.Set(out, Downcast<AffineType>(vals[1])); // ??
        } catch (const std::exception& e) {
           // std::cout << "ICE exception " << e.what() << std::endl;
        }
      }

    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  ExprSet subgraph_;
  AffineTypeMap affine_types_;
  AffineType out_type_;
  const bool hard_fail_;
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
};

class FakeQuantizationRewriterOne : public MixedModeMutator {
 public:
  explicit FakeQuantizationRewriterOne(bool hard_fail) : hard_fail_(hard_fail) {}
  Expr InferType(const Expr& expr) {
    // std::cout << "InferType" << std::endl;
    auto mod = IRModule::FromExpr(expr);
    mod = transform::InferType()(mod);
    if (expr.as<FunctionNode>()) {
      return mod->Lookup("main");
    } else {
      return mod->Lookup("main").as<FunctionNode>()->body;
    }
  }
  bool mutated = false;
 protected:


  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
     // std::cout << ">>>>>> FQORewriter::Rewrite_ " << std::endl;
    // // // std::cout << "memo_ " << std::endl;
    // for (const auto& it : memo_) {
    //   // // // std::cout << "first " << it.first << std::endl;
    //   // // // std::cout << "second " << it.second << std::endl;
    // }
    // // // std::cout << "/memo_ " << std::endl;

     // std::cout << "pre  " << pre->op << std::endl;
    // // std::cout << "pre  " << GetRef<Expr>(pre) << std::endl;
    if (const CallNode* call_node = post.as<CallNode>()) {
      // std::cout << "post " << call_node->op << std::endl;
      // // std::cout << "post " << post << std::endl;
      static auto fqfq =
          Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
      const Op op = Downcast<Op>(call_node->op);
      if (is_enabled_op(call_node)) {
      //if (/* call_node->op != batch_matmul_op_ &&  */ call_node->op != dequantize_op_ && call_node->op != quantize_op_ && fqfq.count(Downcast<Op>(op))) {
        SubgraphExtractorOne extractor;
        // // std::cout << "GetSubgraph" << std::endl;
        ExprSet subgraph = extractor.GetSubgraph(post);
         //ExprSet subgraph = extractor.GetSubgraph(GetRef<Expr>(pre));
        AffineTypeMap affine_types = extractor.GetAffineTypes();
        // // std::cout << "/GetSubgraph" << std::endl;
        // // std::cout << "subgraph" << std::endl;
        // for (const auto& it : subgraph) {
        //   // std::cout << "k " << it << std::endl << std::endl;
        // }
        // // std::cout << "/subgraph" << std::endl;

        //  // // std::cout << "affine_types" << std::endl;
        // for (const auto& it : affine_types) {
        //    // // std::cout << "k " << it.first << std::endl;
        //    // // std::cout << "v " << it.second << std::endl;
        // }
        //  // // std::cout << "/affine_types" << std::endl;

        // // // std::cout << "memo_ " << std::endl;
        //  for (const auto& it : memo_) {
        //     // // std::cout << "first " << it.first << std::endl;
        //     // // std::cout << "second " << it.second << std::endl;
        //  }
        //  // // std::cout << "/memo_ " << std::endl;

        ExprSet post_subgraph;
        AffineTypeMap post_affine_types;
        // // // std::cout << "make post_affine_types" << std::endl;
        for (auto kv : affine_types) {
          if (pre == kv.first.as<CallNode>()) {
            // we havent memoized the current op yet
             // // std::cout << "pre == kv.first.as<CallNode>() " << post << std::endl;
            post_affine_types.Set(post, kv.second);
          } else {
            //auto it = this->memo_.find(kv.first);
            //if (it != memo_.end()) {
            //} else {
            //  memo_[kv.first] = kv.first;
            //}
             auto it = this->memo_.find(kv.first);
             if (it != memo_.end()) {
               post_affine_types.Set(memo_.at(kv.first), kv.second);
            } else {
               post_affine_types.Set(kv.first, kv.second);
            }
             // // std::cout << "pre != kv.first.as<CallNode>() " << kv.first << std::endl;
             // // std::cout << "pre != kv.first.as<CallNode>() memo " << memo_.at(kv.first) << std::endl;
            /*post_affine_types.Set(memo_.at(kv.first), kv.second);*/
          }
        }
        // // // std::cout << "/make post_affine_types" << std::endl;
        // // // std::cout << "make post_subgraph" << std::endl;
        for (auto expr : subgraph) {
            //// std::cout << "insert " << expr << std::endl;
          // // // std::cout << "insert memo " << memo_[expr] << std::endl;
          auto it = this->memo_.find(expr);
          //if (it != memo_.end()) {
          //} else {
          //  memo_[expr] = expr;
          //}
           if (it != memo_.end()) {
             post_subgraph.insert(memo_.at(expr));
          } else {
             post_subgraph.insert(expr);
          }
          /*post_subgraph.insert(memo_.at(expr));*/
          //post_subgraph.insert(memo_[expr]);
        }
        // // // std::cout << "/make post_subgraph" << std::endl;
        // // // std::cout << "post_subgraph" << std::endl;
        // for (const auto& it : post_subgraph) {
        //   // // // std::cout << "k " << it << std::endl << std::endl;
        // }
        // // // std::cout << "/post_subgraph" << std::endl;

        // // // std::cout << "post_affine_types" << std::endl;
        // for (const auto& it : post_affine_types) {
        //   // // // std::cout << "k " << it.first << std::endl;
        //   // // // std::cout << "v " << it.second << std::endl;
        // }
        // // // std::cout << "/post_affine_types" << std::endl;
        // std::cout << "MutateSubgraph" << std::endl;
        Expr out =
            SubgraphMutatorOne(post_subgraph, post_affine_types, hard_fail_).MutateSubgraph(post);
        // std::cout << "/MutateSubgraph" << std::endl;
          // std::cout << "out           " << out << std::endl;

        //if (out != post) {
          //out = InferType(out);
          //out = tvm::relay::transform::InferTypeLocal(out);
           //// std::cout << "out InferType " << out << std::endl;
        //}
        //if (!mutated) {
        //  mutated = (out != post);
        //}
        return out;
      }
    }
    return post;
  }
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  //const Op batch_matmul_op_ = Op::Get("qnn.batch_matmul");
  const bool hard_fail_;
};

/* Description of FakeQuantizationToInteger
 *
 * The purpose of this pass is to find regions of the graph that follow
 * the general pattern:
 *
 *   x    w
 *   |    |
 *   dq   dq
 *    \   /
 *     op1
 *      |
 *     op2
 *      |
 *      q
 *
 * and convert them into subgraphs with actual integer operations on x and w
 *
 * The pass does this via a multi-pass approach:
 *
 * The main pass is a MixedModeMutator that traverses the full graph searching for
 * quantize operations
 *
 * The second pass is an ExprVisitor that recursively searches for subgraphs leading to the
 * quantize for subtraphs bounded by dequantize operations. This pass extracts the affine
 * types of the inputs for later processing, where affine denotes the transformation
 * x_real = (x_affine - zero_point) * scale
 *
 * The third pass is an ExprMutator that recursively rewrites the subgraphs using packed funcs
 * registered with the FTVMFakeQuantizationToInteger attribute. These packed funcs rewrite
 * the ops based on the affine types of their inputs and then return the affine types of the
 * new rewriten ops to pass that information down the stack during rewrite.
 *
 * After the second and third passes run, the first pass replaces the quantize with the
 * rewritten subgraph and the processing continues
 */

// std::string to_expr_str(const Expr& expr) {
//  if (const CallNode* call_node = expr.as<CallNode>()) {
//    std::ostringstream os;
//    os << call_node->op;
//    return os.str();
//  }
//  return "<null>";
//}

class SubgraphExtractor : public ExprVisitor {
 public:
  const ExprSet GetSubgraph(const Expr& expr) {
    std::string expr_str = to_expr_str(expr);
    // // std::cout << ">>>>>> SE::GetSubgraph " << expr_str << std::endl;

    VisitExpr(expr);
    ExprSet subgraph;
    // // std::cout << "collect subgraph" << std::endl;
    if (is_fake_quantized_) {
      for (auto kv : this->visit_counter_) {
        if (auto call_node = GetRef<ObjectRef>(kv.first).as<CallNode>()) {
          // // std::cout << "k " << call_node->op << std::endl;
          // // std::cout << "v " << kv.second << std::endl;
          if (call_node->op != quantize_op_) {
            subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
          }
        }
      }
    }
    // // std::cout << "/collect subgraph" << std::endl;
    return subgraph;
  }
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
    // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
    // abort the rewrite.
    std::string expr_str = to_expr_str(expr);
    // // std::cout << ">>>>>> SE::VisitExpr " << expr_str << std::endl;
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      // // std::cout << "FakeQuantizationToInteger found a non-dataflow op inside a fake quantize region, aborting this rewrite" << std::endl;
      DLOG(INFO) << "FakeQuantizationToInteger found a non-dataflow op inside"
                 << " a fake quantize region, aborting this rewrite";
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    // // std::cout << ">>>>>> SE::VisitExpr_ " << call_node->op << std::endl;
    if (call_node->op == quantize_op_) {
      const auto* attrs = call_node->attrs.as<qnn::QuantizeAttrs>();
      ICHECK(attrs != nullptr);
      // Only look at arg0 for quantize
      VisitExpr(call_node->args[0]);
      // Collect type of quantize ops
      affine_types_.Set(
          GetRef<Expr>(call_node),
          TensorAffineType(call_node->args[1], call_node->args[2], attrs->out_dtype, attrs->axis));
    } else if (call_node->op == dequantize_op_) {
      const auto* attrs = call_node->attrs.as<qnn::DequantizeAttrs>();
      ICHECK(attrs != nullptr);
      // VisitExpr(call_node->args[0]);// for abs
      // Collect type of dequantize ops
      affine_types_.Set(
          GetRef<Expr>(call_node),
          TensorAffineType(call_node->args[1], call_node->args[2],
                           call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype,
                           attrs->axis));
    } else {
      // run normally on everything else.
      ExprVisitor::VisitExpr_(call_node);
    }
  }

  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
};

class SubgraphMutator : public ExprMutator {
 public:
  SubgraphMutator(ExprSet subgraph, AffineTypeMap affine_types, bool hard_fail)
      : subgraph_(subgraph), affine_types_(affine_types), hard_fail_(hard_fail) {}

  Expr MutateSubgraph(const Expr& expr) {
    std::string expr_str = to_expr_str(expr);
    // // std::cout << ">>>>>> SM::MutateSubgraph " << expr_str << std::endl;
    if (subgraph_.size() == 0) {
      return expr;
    }
    const CallNode* quantize_node = expr.as<CallNode>();
    ICHECK(quantize_node);
    ICHECK(quantize_node->op == quantize_op_);
    out_type_ = affine_types_[expr];
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    // // std::cout << "subgraph_" << std::endl;

    for (auto node : subgraph_) {
      const Op op = Downcast<Op>(node.as<CallNode>()->op);
      if (!fqfq.count(Downcast<Op>(op))) {
        // Only modify the subgraph if we have translation
        // rules for every op
        if (hard_fail_) {
          LOG(FATAL) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
        } else {
          DLOG(INFO) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
          return expr;
        }
      }
    }
    try {
      return Mutate(expr);
    } catch (std::exception& e) {
      if (hard_fail_) {
        throw e;
      } else {
        DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping" << expr << std::endl;
        return expr;
      }
    }
  }

 protected:
  Expr VisitExpr_(const CallNode* call_node) {
    // // std::cout << ">>>>>> SM::VisitExpr_ " << call_node->op << std::endl;
    Expr out;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    Op op = Downcast<Op>(call_node->op);
    if (fqfq.count(op)) {
      Expr expr;
      if (op == dequantize_op_) {
        expr = GetRef<Expr>(call_node);
      } else {
        expr = ExprMutator::VisitExpr_(call_node);
        // Set the current op to the output type, useful if we can't deduce output parameters
        // from input parameters
        affine_types_.Set(expr, out_type_);
      }
      // Call the rewrite
      Array<ObjectRef> vals = fqfq[op](expr, affine_types_);
      // Save the outputs of the rewrite
      ICHECK(vals.size() == 2)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);
      affine_types_.Set(out, Downcast<AffineType>(vals[1]));
    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  ExprSet subgraph_;
  AffineTypeMap affine_types_;
  AffineType out_type_;
  const bool hard_fail_;
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
};

class FakeQuantizationRewriter : public MixedModeMutator {
 public:
  explicit FakeQuantizationRewriter(bool hard_fail) : hard_fail_(hard_fail) {}

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    // // std::cout << ">>>>>> FQRewriter::Rewrite_ " << std::endl;
    // // std::cout << "memo_ " << std::endl;
    //for (const auto& it : memo_) {
      // // std::cout << "first " << it.first << std::endl;
      // // std::cout << "second " << it.second << std::endl;
    //}
    // // std::cout << "/memo_ " << std::endl;
    // // std::cout << "pre " << pre->op << std::endl;
    if (const CallNode* call_node = post.as<CallNode>()) {
      // // std::cout << "post " << call_node->op << std::endl;
      if (call_node->op == quantize_op_) {
        SubgraphExtractor extractor;
        // // std::cout << "GetSubgraph" << std::endl;
        ExprSet subgraph = extractor.GetSubgraph(GetRef<Expr>(pre));
        AffineTypeMap affine_types = extractor.GetAffineTypes();
        // // std::cout << "/GetSubgraph" << std::endl;
        // // std::cout << "subgraph" << std::endl;
        // for (const auto& it : subgraph) {
        //   // // std::cout << "k " << it << std::endl << std::endl;
        // }
        // // std::cout << "/subgraph" << std::endl;

        // // // std::cout << "affine_types" << std::endl;
        // for (const auto& it : affine_types) {
        //   // // std::cout << "k " << it.first << std::endl;
        //   // // std::cout << "v " << it.second << std::endl;
        // }
        // // std::cout << "/affine_types" << std::endl;

        ExprSet post_subgraph;
        AffineTypeMap post_affine_types;
        // // std::cout << "make post_affine_types" << std::endl;
        for (auto kv : affine_types) {
          if (pre == kv.first.as<CallNode>()) {
            // we havent memoized the current op yet
            // // std::cout << "pre == kv.first.as<CallNode>() " << post << std::endl;
            post_affine_types.Set(post, kv.second);
          } else {
            // // std::cout << "pre != kv.first.as<CallNode>() " << kv.first << std::endl;
            // // std::cout << "pre != kv.first.as<CallNode>() memo " << memo_.at(kv.first) << std::endl;
            post_affine_types.Set(memo_.at(kv.first), kv.second);
          }
        }
        // // std::cout << "/make post_affine_types" << std::endl;
        // // std::cout << "make post_subgraph" << std::endl;
        for (auto expr : subgraph) {
          // std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
          // // std::cout << "insert " << expr << std::endl;
          // // std::cout << "insert memo " << memo_[expr] << std::endl;
          post_subgraph.insert(memo_[expr]);
        }
        // // std::cout << "/make post_subgraph" << std::endl;
        // // std::cout << "post_subgraph" << std::endl;
        // for (const auto& it : post_subgraph) {
        //   // // std::cout << "k " << it << std::endl << std::endl;
        // }
        // // std::cout << "/post_subgraph" << std::endl;

        // // std::cout << "post_affine_types" << std::endl;
        // for (const auto& it : post_affine_types) {
        //   // // std::cout << "k " << it.first << std::endl;
        //   // // std::cout << "v " << it.second << std::endl;
        // }
        // // std::cout << "/post_affine_types" << std::endl;
        // // std::cout << "MutateSubgraph" << std::endl;
        Expr out =
            SubgraphMutator(post_subgraph, post_affine_types, hard_fail_).MutateSubgraph(post);
        // // std::cout << "/MutateSubgraph" << std::endl;

        // // std::cout << "out " << out << std::endl;
        

        return out;
      }
    }
        
    return post;
  }
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const bool hard_fail_;
};

Expr FakeQuantizationToInteger(const Expr& expr, const IRModule& mod, bool hard_fail) {
  auto before_expr = FakeQuantizationRewriter(hard_fail).Mutate(expr);
   //// std::cout << "Mutate(expr) " << to_expr_str(expr) << std::endl;

  // std::cout << "FakeQuantizationRewriter " << before_expr << std::endl;
  // std::cout << "####################################################################" << std::endl;
  // std::cout << "####################################################################" << std::endl;
  // std::cout << "####################################################################"<< std::endl;

  auto pass = FakeQuantizationRewriterOne(hard_fail);


  // std::cout << "InferType " << before_expr << std::endl;
  // std::cout << "####################################################################" << std::endl;
  // std::cout << "####################################################################" << std::endl;
  // std::cout << "####################################################################" << std::endl;
  before_expr = pass.InferType(before_expr);



  //const Expr& before_expr = expr;
  auto after_expr = pass.Mutate(before_expr);



  //int i = 1;
  //while (pass.mutated /* && i < 10*/) {
  ////while (before_expr != after_expr) {
  // // std::cout << "inter: " << i << std::endl;
  // after_expr = pass.InferType(after_expr);
  // pass.mutated = false;
  // after_expr = pass.Mutate(after_expr);
  // i++;
  //}
  // auto rewritten_expr = before_expr;
  auto rewritten_expr = after_expr;


  

  // auto rewritten_expr = FakeQuantizationRewriter(hard_fail).Mutate(expr);
  // // // std::cout << "(expr == rewritten_expr) " << std::boolalpha << (expr == rewritten_expr) <<
  // std::endl; if (expr == rewritten_expr)
  //  return FakeQuantizationRewriterOne(hard_fail).Mutate(rewritten_expr);
  // else
  return rewritten_expr;
}

namespace transform {

Pass FakeQuantizationToInteger(bool hard_fail) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FakeQuantizationToInteger(f, m, hard_fail));
      };
  return CreateFunctionPass(pass_func, 0, "FakeQuantizationToInteger", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FakeQuantizationToInteger")
    .set_body_typed(FakeQuantizationToInteger);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
