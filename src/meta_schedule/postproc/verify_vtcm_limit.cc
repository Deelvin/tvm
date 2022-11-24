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
#include <tvm/tir/transform.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief Verify the correctness of the generated Hexagon code. */
class VerifyVTCMLimitNode : public PostprocNode {
 public:
  int64_t vtcm_capacity = 0;

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();
    vtcm_capacity = target->GetAttr<Integer>("vtcm-capacity").value().IntValue();
  }

  bool Verify(const IRModule& mod) const {
    for (const auto& kv : mod->functions) {
      if (auto* n = kv.second.as<tir::PrimFuncNode>()) {
        auto func = GetRef<tir::PrimFunc>(n);
        auto sizes = CalculateAllocatedBytes(func);
        const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
        if (vtcm_capacity > 0 && vtcm_allocated.IntValue() > vtcm_capacity) {
          return false;
        }
      }
    }
    return true;
  }

  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
        IRModule lowered{nullptr};
        try {
          auto pass_list = Array<tvm::transform::Pass>();
          // Convert Function to IRModule
          transform::PassContext pass_ctx = transform::PassContext::Current();
          tir::PrimFunc f = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol",
                                     runtime::String(g_var->name_hint));
          bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
          if (noalias) {
            f = WithAttr(std::move(f), "tir.noalias", Bool(true));
          }
          IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
        } catch (const dmlc::Error& e) {
          return false;
        }
        if (!Verify(lowered)) {
          return false;
        }
      }
    }
    return true;
  }

  Postproc Clone() const {
    ObjectPtr<VerifyVTCMLimitNode> n = make_object<VerifyVTCMLimitNode>(*this);
    n->vtcm_capacity = this->vtcm_capacity;
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.VerifyVTCMLimit";
  TVM_DECLARE_FINAL_OBJECT_INFO(VerifyVTCMLimitNode, PostprocNode);
};

Postproc Postproc::VerifyVTCMLimit() {
  ObjectPtr<VerifyVTCMLimitNode> n = make_object<VerifyVTCMLimitNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(VerifyVTCMLimitNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyVTCMLimit")
    .set_body_typed(Postproc::VerifyVTCMLimit);

}  // namespace meta_schedule
}  // namespace tvm
