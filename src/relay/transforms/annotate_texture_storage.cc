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
 * \file annotate_texture_storage.cc
 * \brief Collection of target specific relay passes which
 * storage scope related information.
 *
 *  - CollectStorageInfo returns a mapping from relay expr
 *    to a list of output storage scopes for each output.
 *    These scopes are used during memory planning as well
 *    as downstream when doing codegen and in the graph runtime when doing runtime dataspace
 *    allocations.
 *
 *  - AnnotateMemoryScope calls *target.CollectStorageInfo for all target been represented
 *    in the graph and rewrites graph modifying or inserting of VirtualDevice with required
 *    memory_scop collected from the CollectStorageInfo
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/expr.h>

#include <memory>
#include <unordered_map>

#include "../transforms/device_aware_visitors.h"

namespace tvm {
namespace relay {
namespace {

/**
 * @brief Analyzes the graph and returns mapping of expressions vs desired memory scope
 */
class StorageInfo : private transform::DeviceAwareExprVisitor {
 public:
  StorageInfo() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}

  static Map<Expr, Array<String>> GetStorageMap(const Expr& expr) {
    StorageInfo storage_info;
    storage_info.VisitExpr(expr);
    storage_info.LegalizeProducerStorage();
    // For now we force write to global for the outputs of the function over which
    // memory planning will be performed. This should incur only a trivial change
    // in performance.
    storage_info.ForceGlobalOutputStorage(expr);
    Map<Expr, Array<String>> storage_map;
    for (auto& kv : storage_info.storage_scope_) {
      if (storage_info.constants_expr_.find(GetRef<Expr>(kv.first)) !=
          storage_info.constants_expr_.end()) {
        std::vector<String> storage_scopes;
        std::copy(kv.second.begin(), kv.second.end(), std::back_inserter(storage_scopes));
        storage_map.Set(GetRef<Expr>(kv.first), Array<String>{storage_scopes});
      }
    }

    // initial algo assumes mapping of outputs of the expr that is not enough, need to update
    // VirtualDevice for function variables to get proper codegen. Adding vars to storage_map
    for (const auto& a : storage_info.args_to_vars_) {
      if (storage_map.count(a.first)) {
        storage_map.Set(a.second, storage_map[a.first]);
      }
    }
    return storage_map;
  }

 private:
  void Visit(const Expr& expr) {
    // Pre-order traversal to enable upward propagation
    // of consumer storage scopes to producers when desirable.
    if (const auto* fn = expr.as<FunctionNode>()) {
      this->VisitExpr(fn->body);
      for (const auto& param : fn->params) {
        this->VisitExpr(param);
      }
    } else {
      this->VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* vn) final { ApplyConsumerScopeToInputs(vn); }

  void VisitExpr_(const ConstantNode* cn) final {
    constants_expr_.insert(GetRef<Expr>(cn));
    ApplyConsumerScopeToInputs(cn);
  }

  void DeviceAwareVisitExpr_(const CallNode* call) final {
    // Check the contents of this primitive function
    if (DeviceSupportsTextureStorage(GetRef<Expr>(call))) {
      if (const auto* fn = call->op.as<FunctionNode>()) {
        if (fn->HasNonzeroAttr(attr::kPrimitive)) {
          primitive_supports_texture_ = false;
          Visit(call->op);
          if (primitive_supports_texture_) {
            if (call->checked_type().as<TensorTypeNode>()) {
              std::string scope = "global.texture";
              if (const auto* ttype = call->checked_type().as<TensorTypeNode>()) {
                if (ttype->shape.size() == 5) {
                  scope = Scope(ttype->shape);
                }
              }
              storage_scope_[call].push_back(scope);
            } else {
              const auto* tuple_type = call->type_as<TupleTypeNode>();
              ICHECK(tuple_type);
              // TODO(csullivan): Add support for mixed output storage scope.
              // In current adreno storage planner all outputs of a
              // primitive function are assumed to be of the same storage
              // type. This should be easy to extend in the future.
              for (size_t i = 0; i < tuple_type->fields.size(); i++) {
                storage_scope_[call].push_back("global.texture");
              }
            }
            for (size_t i = 0; i < fn->params.size(); i++) {
              args_to_vars_[call->args[i]] = fn->params[i];
            }
          }
          // Add consumer storage scope information for call arguments
          for (auto& arg : call->args) {
            if (storage_scope_.count(call)) {
              ICHECK(!HasMixedStorageOutputs(call))
                  << "Mixed output storage scopes are not currently supported";
              consumer_storage_scopes_[arg.operator->()].push_back(storage_scope_[call][0]);
            } else {
              consumer_storage_scopes_[arg.operator->()].push_back("global");
            }
          }
        }
      }
    }

    primitive_supports_texture_ = SupportsTextureStorage(call);

    for (auto& arg : call->args) {
      Visit(arg);
    }
  }

  std::string Scope(Array<PrimExpr> shape) {
    std::map<int, std::string> diffs;
    int limit = 16384;
    int a0 = shape[0].as<IntImmNode>()->value;
    int a1 = shape[1].as<IntImmNode>()->value;
    int a2 = shape[2].as<IntImmNode>()->value;
    int a3 = shape[3].as<IntImmNode>()->value;

    int d3l = a0 * a1 * a2;
    int d3r = a3;
    int diff3 = d3l > d3r ? d3l - d3r : d3r - d3l;
    if (d3l < limit && d3r < limit) diffs[diff3] = "";

    int d2l = a0 * a1;
    int d2r = a2 * a3;
    int diff2 = d2l > d2r ? d2l - d2r : d2r - d2l;
    if (d2l < limit && d2r < limit) diffs[diff2] = "nhwc";

    int d1l = a0;
    int d1r = a1 * a2 * a3;
    int diff1 = d1l > d1r ? d1l - d1r : d1r - d1l;
    if (d1l < limit && d1r < limit) diffs[diff1] = "weight";
    if (!diffs.empty()) {
      std::string scope = "global.texture";
      if (!diffs.begin()->second.empty()) {
        scope += ("-" + diffs.begin()->second);
      }
      return scope;
    } else {
      return "global.texture";
    }
  }

  void ApplyConsumerScopeToInputs(const ExprNode* expr) {
    std::string scope;
    auto consumer_scopes_it = consumer_storage_scopes_.find(expr);
    if (consumer_scopes_it != consumer_storage_scopes_.end()) {
      std::string consumer_scope = GetConsumerScope(consumer_scopes_it->second);
      ICHECK(!storage_scope_.count(expr))
          << "Already propagated consumer scopes to input: " << GetRef<Expr>(expr);

      bool expr_is_rgba_vectorizable = false;
      if (const auto* ttype = expr->checked_type().as<TensorTypeNode>()) {
        if (ttype->shape.size() == 5) {
          scope = Scope(ttype->shape);
          if (scope != "global") {
            auto inner_dim = ttype->shape.back().as<IntImmNode>();
            if (inner_dim && inner_dim->value == 4) {
              expr_is_rgba_vectorizable = true;
            }
          }
        }
      }

      // Only propagate texture scope from consumers to input expr if
      // the input shape of the input expr is rgba vectorizable.
      if (consumer_scope.find("global.texture") != std::string::npos) {
        if (expr_is_rgba_vectorizable) {
          storage_scope_[expr].push_back(scope);
        }
      } else {
        storage_scope_[expr].push_back(consumer_scope);
      }
    }
  }

  void LegalizeProducerStorage() {
    for (auto& kv : consumer_storage_scopes_) {
      const ExprNode* producer = kv.first;
      std::string legal_scope = GetConsumerScope(kv.second);
      if (storage_scope_.count(producer)) {
        ICHECK(!HasMixedStorageOutputs(producer))
            << "Mixed output storage scopes are not currently supported";
        if (storage_scope_[producer][0].find(legal_scope) == std::string::npos) {
          for (size_t i = 0; i < storage_scope_[producer].size(); i++) {
            // Only support uniform storage scope across all outputs for now
            storage_scope_[producer][i] = legal_scope;
          }
        }
      }
    }
  }

  void ForceGlobalOutputStorage(const Expr& expr) {
    // Mark function outputs as global scope
    if (const auto* func = expr.as<FunctionNode>()) {
      if (auto* tuple = func->body.as<TupleNode>()) {
        for (auto& field : tuple->fields) {
          if (storage_scope_.count(field.operator->())) {
            for (size_t i = 0; i < storage_scope_[field.operator->()].size(); i++) {
              storage_scope_[field.operator->()][i] = "global";
            }
          }
        }
      } else {
        if (storage_scope_.count(func->body.operator->())) {
          for (size_t i = 0; i < storage_scope_[func->body.operator->()].size(); i++) {
            storage_scope_[func->body.operator->()][i] = "global";
          }
        }
      }
    }
  }

  bool DeviceSupportsTextureStorage(const Expr& expr) {
    auto vd = GetVirtualDevice(expr);
    if (vd != VirtualDevice::FullyUnconstrained()) {
      if (Optional<String> t_device = vd->target->GetAttr<String>("device")) {
        if (vd->target->kind->device_type == kDLOpenCL && t_device.defined()) {
          if (t_device.value() == "adreno") {
            return true;
          }
        }
      }
    }
    return false;
  }

  std::string GetConsumerScope(const std::vector<std::string>& consumer_scopes) const {
    if (!consumer_scopes.size()) {
      return "global";
    }
    std::string texture_tag = "global.texture";
    for (auto& consumer_scope : consumer_scopes) {
      if (consumer_scope.find(texture_tag) == std::string::npos) {
        return "global";
      }
    }
    return texture_tag;
  }

  bool HasMixedStorageOutputs(const ExprNode* expr) {
    if (storage_scope_.count(expr)) {
      std::string ref_scope = storage_scope_[expr][0];
      for (std::string& scope : storage_scope_[expr]) {
        if (scope != ref_scope) {
          return true;
        }
      }
    }
    return false;
  }

  bool SupportsTextureStorage(const CallNode* call) const {
    bool supports_texture_storage = false;
    if (auto attrs = call->attrs.as<Conv2DAttrs>()) {
      if (attrs->data_layout == "NCHW4c" && attrs->kernel_layout == "OIHW4o") {
        supports_texture_storage = true;
      } else if (attrs->data_layout == "NHWC4c" &&
                 (attrs->kernel_layout == "HWOI4o" || attrs->kernel_layout == "HWIO4o" ||
                  attrs->kernel_layout == "OIHW4o")) {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<GlobalPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<MaxPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<AvgPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    }

    return supports_texture_storage;
  }

  /*! \brief Temporary state for marking whether a visited function
   *         primitive supports texture storage scope */
  bool primitive_supports_texture_ = false;
  /*! \brief expr storage scope mapping for each output  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> storage_scope_;
  /*! \brief output storage scopes used by consumers of expr key  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> consumer_storage_scopes_;
  /*! \brief mapping of arguments to call to function variables*/
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> args_to_vars_;
  /*! \brief set of constant expressions to filter out expressions*/
  std::set<Expr> constants_expr_;
};

}  // namespace

/**
 * @brief rewrite of virtual devices, memory_scope part for expressions defined
 * by the StorageInfo analysis pass
 *
 * Currently this workflow supports analysis and rewriting of VirtualDevice for
 * Constants and function Variables
 */
class VDRewriter : public transform::DeviceAwareExprMutator {
  using VarMap = std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual>;

 public:
  explicit VDRewriter(const Map<Expr, Array<String>>& storage_scope)
      : transform::DeviceAwareExprMutator(Optional<IRModule>()), storage_scope_(storage_scope) {}

  Function Rewrite(const Expr& expr) { return Downcast<Function>(Mutate(expr)); }

  Expr VisitExpr_(const VarNode* vn) final {
    if (storage_scope_.find(GetRef<Expr>(vn)) != storage_scope_.end() &&
        storage_scope_[GetRef<Expr>(vn)][0] != "global") {
      Var c = Var(vn->vid, vn->type_annotation, vn->span);
      auto virtual_device = GetVirtualDevice(GetRef<Expr>(vn));
      c->virtual_device_ =
          VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                        virtual_device->target, storage_scope_[GetRef<Expr>(vn)][0]);
      return c;
    }
    return GetRef<Var>(vn);
  }

  Expr VisitExpr_(const ConstantNode* vn) final {
    if (storage_scope_.find(GetRef<Expr>(vn)) != storage_scope_.end() &&
        storage_scope_[GetRef<Expr>(vn)][0] != "global") {
      Expr c = Constant(vn->data, vn->span);
      auto virtual_device = GetVirtualDevice(GetRef<Expr>(vn));
      c = OnDevice(c,
                   VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                                 virtual_device->target, storage_scope_[GetRef<Expr>(vn)][0]),
                   true);
      return c;
    }
    return GetRef<Constant>(vn);
  }

 private:
  Map<Expr, Array<String>> storage_scope_;
  VarMap new_vars_;
  Array<String> current_function_scope_;
};

Map<Expr, Array<String>> CollectTextureStorage(const Expr& expr) {
  return StorageInfo::GetStorageMap(expr);
}

/**
 * @brief Collects all target devices participated in graph
 */
class CollectVirtualDevices : public transform::DeviceAwareExprVisitor {
 public:
  CollectVirtualDevices() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}
  /**
   * @brief Get all unique device elements from target of each VirtualDevice
   *
   * @param expr - IR
   * @return set of devices
   */
  std::set<std::string> GetDevices(const Expr& expr) {
    this->Run(expr);
    return std::move(devices_);
  }

  void Visit(const Expr& expr) {
    // Pre-order traversal to enable upward propagation
    // of consumer storage scopes to producers when desirable.
    if (const auto* fn = expr.as<FunctionNode>()) {
      this->VisitExpr(fn->body);
      for (const auto& param : fn->params) {
        this->VisitExpr(param);
      }
    } else {
      this->VisitExpr(expr);
    }
  }

  void DeviceAwareVisitExpr_(const CallNode* call) final {
    auto vd = GetVirtualDevice(GetRef<Expr>(call));
    if (vd != VirtualDevice::FullyUnconstrained()) {
      if (Optional<String> t_device = vd->target->GetAttr<String>("device")) {
        devices_.insert(vd->target->kind->name + "." + t_device.value());
      }
    }
    for (auto& arg : call->args) {
      Visit(arg);
    }
  }

  void Run(const Expr& expr) { VisitExpr(expr); }
  using transform::DeviceAwareExprVisitor::VisitExpr_;
  std::set<std::string> devices_;
};

/*!
 * \brief Collect the target specific tensor storage info for each expression's output.
 * \param expr The expression.
 * \return The device based storage mapping.
 */
Map<Expr, Array<String>> CollectStorageInfo(const Expr& expr) {
  std::set<std::string> device_types = CollectVirtualDevices().GetDevices(expr);
  // TODO(amalyshe): current approach collects all targets withing graph and call the only
  // function corresponding to all these targets in alphabetic order
  // this will work reliable only for case of only one device and should be redesigned
  // to handle common case
  std::string ftarget_prefix = "relay.backend";
  for (auto& dev_id : device_types) {
    ftarget_prefix += (std::string(".") + dev_id);
  }

  Map<Expr, Array<String>> storage_info = {};
  if (const auto* f = runtime::Registry::Get(ftarget_prefix + "._CollectStorageInfo")) {
    storage_info = (*f)(expr);
  }
  return storage_info;
}

Expr AnnotateMemoryScopeExpr(const Expr& expr, const IRModule& mod, CompilationConfig config) {
  auto storage_scope = CollectStorageInfo(expr);
  return VDRewriter(storage_scope).Rewrite(expr);
}

namespace transform {
tvm::transform::Pass AnnotateMemoryScope(CompilationConfig config) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [config = std::move(config)](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(AnnotateMemoryScopeExpr(f, m, config));
      };
  return CreateFunctionPass(pass_func, 2, "AnnotateMemoryScope", {});
}
}  // namespace transform

TVM_REGISTER_GLOBAL("relay.backend.opencl.adreno._CollectStorageInfo")
    .set_body_typed(CollectTextureStorage);

}  // namespace relay
}  // namespace tvm
