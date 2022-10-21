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
 * \file tir/analysis/calculate_allocated_memory.cc
 * \brief Calculate allocated memory per memory scope required by PrimFuncs.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/container/map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <map>
#include <algorithm>

namespace tvm {
namespace tir {

template <typename T>
class AllocationCalculator : public StmtExprVisitor {
 public:
  AllocationCalculator() = default;
  std::map<std::string, size_t> operator()(const PrimFunc& func);

 private:
  void VisitStmt_(const T* op) override;
  std::map<std::string, size_t> _max_size;
  std::map<std::string, size_t> _current_size;
};

template <typename T>
std::map<std::string, size_t> AllocationCalculator<T>::operator()(const PrimFunc& func) {
  this->VisitStmt(func->body);
  return this->_max_size;
}

// TODO ICE Delete this copy or make it as utils
std::string GetStorageScope(const Var& var) { 
  auto* ptr = var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
  return ptr->storage_scope;
}

template <typename T>
void AllocationCalculator<T>::VisitStmt_(const T* op) {
  auto alloc_size = op->ConstantAllocationSize();
  std::string storage_scope = GetStorageScope(op->buffer_var);

  auto search = _current_size.find(storage_scope);
  if (search == _current_size.end()) {
      _current_size[storage_scope] = 0;
      _max_size[storage_scope] = 0;
  }

  _current_size[storage_scope] += alloc_size;
  _max_size[storage_scope] = std::max(_current_size[storage_scope], _max_size[storage_scope]);
  StmtExprVisitor::VisitStmt(op->body);
  _current_size[storage_scope] -= alloc_size;
}


std::map<std::string, size_t> CalculateAllocatedBytes(const PrimFunc& func) {
  return AllocationCalculator<AllocateNode>()(func);
}

TVM_REGISTER_GLOBAL("tir.analysis.calculate_allocated_bytes")
    .set_body_typed([](PrimFunc func) {
      auto sizes = CalculateAllocatedBytes(func);
      tvm::Map<String, Integer> res;
      for(auto [k, v] : sizes) {
        res.Set(String(k), Integer(v));
      }
      return res;
    });


namespace transform {

Pass VerifyVTCMLimit(size_t limit) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        auto sizes = CalculateAllocatedBytes(func);
        static const char vtcm_name[] = "global.vtcm";
        auto search = sizes.find(vtcm_name);
        size_t vtcm_allocated = search != sizes.end() ? sizes[vtcm_name] : 0;

        std::cout << "VerifyVTCMLimit" << " global.vtcm memory allocation (allocated: "
            << vtcm_allocated << ", limit: " << limit << ")" << std::endl << std::flush;
        
        if (limit > 0 && vtcm_allocated > limit) {
          LOG(FATAL) << "RuntimeError: The global.vtcm memory allocation limit has been exceeded(allocated: "
            << vtcm_allocated << ", limit: " << limit<< ").\n" << "In function\n" << func;
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.calculate_allocated_bytes", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifyVTCMLimit").set_body_typed(VerifyVTCMLimit);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
