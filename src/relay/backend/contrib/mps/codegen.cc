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
 * \file
 * \brief Implementation of MPS codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <numeric>
#include <sstream>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief Retrieve the expected "root" op nested inside a fused call, such as conv2d in
 *        relu(add(conv2d))
 * \param call A Relay call node. Typically nn.relu when called the first time.
 * \param max_depth The maximum number of calls before the root op, counting from current_call.
 * \param root_name The name of expected "root" op in this fused call.
 * \return A CallNode corresponding to the root op
 */
inline const CallNode* FindCallWithName(const CallNode* current_call, int max_depth,
                                        const std::string& root_name) {
  ICHECK(current_call && max_depth >= 0);

  if (max_depth == 0) {
    ICHECK(current_call && IsOp(current_call, root_name));
    return current_call;
  }
  if (IsOp(current_call, root_name)) {
    return current_call;
  }

  ICHECK_GT(current_call->args.size(), 0);

  const auto* next_call = current_call->args[0].as<CallNode>();
  return FindCallWithName(next_call, max_depth - 1, root_name);
}

class MPSJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  MPSJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    const CallNode* call = cn;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      ICHECK(comp.defined()) << "MPS JSON runtime only supports composite functions.";
      name = comp.value();

      auto body = fn->body.as<CallNode>();
      if (name == "mps.dense") {
          std::cout << " >>> MPS dense!" << std::endl;
        //auto add_op_type = IsOp(body, "add") ? "add" : "nn.bias_add";
        //call = GetRootCall(body, 1, {"nn.conv2d"});
        call = body;
      } else {
        LOG(FATAL) << "Unrecognized MPS pattern: " << name;
      }
    } else {
      LOG(FATAL) << "MPS JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(cn));
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module MPSCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  MPSJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.MPSJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.mps").set_body_typed(MPSCompiler);

/**
 * \brief A helper to expand the params by adding ones which used by MPS runtime
 * for a given expression. Same as default ConstantUpdater but skip constant from
 * essential MPS composed function ops.
 */
struct MPSConstantUpdater : public ConstantUpdater {
 public:
  MPSConstantUpdater(const std::string& symbol,
                      std::unordered_map<std::string, runtime::NDArray>* params,
                      const std::vector<std::string>& skip_mask)
      : ConstantUpdater(symbol, params), skip_mask_(skip_mask) {}
  using ConstantUpdater::VisitExpr_;

  /**!
   * Like an original implementation but avoid visiting of body nodes
   * for MPS specific composite primitives.
   */
  void VisitExpr_(const FunctionNode* op) final {
    this->VisitSpan(op->span);
    for (auto param : op->params) {
      this->VisitExpr(param);
    }

    if (!isMPSSSpecificCompositeFunc(op)) {
      this->VisitExpr(op->body);
    }
  }

 private:
  bool isMPSSSpecificCompositeFunc(const FunctionNode* op) {
    auto comp = op->GetAttr<String>(attr::kComposite);
    if (!comp) return false;

    auto comp_name = comp.value();

    bool is_match = false;
    for (const auto& mask : skip_mask_) {
      if (std::string(comp_name).substr(0, mask.size()) == mask) {
        is_match = true;
        break;
      }
    }
    return is_match;
  }

  std::vector<std::string> skip_mask_;
};

Map<String, runtime::NDArray> MPSConstantUpdaterFunc(Expr expr, std::string symbol) {
  std::vector<std::string> mps_composite_filter = {"mps."};

  // Visit all suitable constant nodes
  std::unordered_map<std::string, runtime::NDArray> res;
  MPSConstantUpdater const_updater(symbol, &res, mps_composite_filter);
  const_updater(expr);

  // Convert to tvm::Map
  Map<String, runtime::NDArray> ret;
  for (const auto& kvp : res) ret.Set(kvp.first, kvp.second);
  return ret;
}

TVM_REGISTER_GLOBAL("relay.ext.mps.constant_updater").set_body_typed(MPSConstantUpdaterFunc);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

