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
 * \file tir/analysis/calculate_inout_tensors.cc
 * \brief Calculate the input and output tensors memory size required by PrimFuncs.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {

template <typename T>
class InOutCalculator : public StmtExprVisitor {
 public:
  InOutCalculator() = default;
  size_t operator()(const PrimFunc& func);
};

static Integer CalculateExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents) {
  size_t element_size_bytes = dtype.bytes();
  size_t num_elements = 1;
  for (const auto& ext : extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    } else {
      // We can't statically calculate workspace for dynamic shapes
      return Integer();
    }
  }
  return Integer(num_elements * element_size_bytes);
}

template <typename T>
size_t InOutCalculator<T>::operator()(const PrimFunc& func) {
  for (size_t i = 0; i < func->params.size(); i++) {
    // func->params[i]
    // std::cout << "func->params[i] " << i << " " <<  func->params[i] << std::flush << std::endl;
    // std::cout << "func->params[i]->dtype " << func->params[i]->dtype.bytes() << std::flush << std::endl;
    // CHECK(func->params[i]->dtype.is_handle()) << "ValueError: Parameters of the description of the "
    //                       "tensor intrinsic should be handle only.";
  }
  size_t size_all = 0;
  for (auto [k,v] : func->buffer_map) {
      Integer num_elements;
      Integer res = Integer();
      num_elements = CalculateExtentsSize(v->dtype, v->shape);
      size_t element_size_bytes = v->dtype.bytes();
      if (!num_elements.defined()) {
        size_all = 0;
        break;
      }
      size_all += num_elements.IntValue();

  }
  return size_all;

}

size_t CalculateIntOutTensorsBytes(const PrimFunc& func) {
  // InOutCalculator<AllocateConstNode> wc;
  // InOutCalculator<AttrStmtNode> wc;
  InOutCalculator<AllocateNode> wc;
  auto size = wc(func);
  return size;
}

TVM_REGISTER_GLOBAL("tir.analysis.calculate_inout_tensors_bytes")
    .set_body_typed([](PrimFunc func) {
      return static_cast<int>(CalculateIntOutTensorsBytes(func));
    });


namespace transform {

Pass VerifySRAMLimit(size_t size_limit) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        auto size = CalculateIntOutTensorsBytes(func);
        // std::cout << "ICE VerifySRAMLimit" << kv.first << " size: " << size << std::flush << std::endl;
        
        if (size_limit != 0 && size > size_limit) {
          LOG(FATAL) << "RuntimeError: calc zero args + return tensors\nIn function\n" << func;
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.calculate_inout_tensors_bytes", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifySRAMLimit").set_body_typed(VerifySRAMLimit);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
