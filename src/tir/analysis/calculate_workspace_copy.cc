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
 * \file tir/analysis/calculate_workspace.cc
 * \brief Calculate any intermediary memory required by PrimFuncs.
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
  // size_t byte_alignment = tvm::runtime::kDefaultWorkspaceAlignment;

 private:
  void VisitStmt_(const T* op) override;
  // size_t GetByteAlignedSize(Integer non_aligned_size);
  // size_t CalculateExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents);
  size_t current_size = 0;
  size_t max_size = 0;
};

template <typename T>
size_t InOutCalculator<T>::operator()(const PrimFunc& func) {
  std::cout << "InOutCalculator::operator" << std::endl;
  std::cout << "func->body\n" << func->body << std::endl;

  // size_t element_size_bytes = dtype.bytes();
  // size_t size = 0;

  for (size_t i = 0; i < func->params.size(); i++) {
    // func->params[i]
    std::cout << "func->params[i] " << i << " " <<  func->params[i] << std::endl;
    std::cout << "func->params[i]->dtype " << func->params[i]->dtype.bytes() << std::endl;
    // CHECK(func->params[i]->dtype.is_handle()) << "ValueError: Parameters of the description of the "
    //                       "tensor intrinsic should be handle only.";
  }
  std::cout << "func->body" << std::endl;
  std::cout << "func->buffer_map.size() " << func->buffer_map.size() << std::endl;



  size_t size_all = 0;
  for (auto [k,v] : func->buffer_map) {
      // v->
      


      size_t num_elements = 1;
      Integer res = Integer();
      for (const auto& ext : v->shape) {
        if (ext->IsInstance<IntImmNode>()) {
          num_elements *= Downcast<IntImm>(ext)->value;
        } else {
          // We can't statically calculate workspace for dynamic shapes
          num_elements = 0;
          break;
        }
      }
      if (num_elements == 0) {
        size_all = 0;
        break;
      }
      
      size_t element_size_bytes = v->dtype.bytes();
      std::cout << "element_size_bytes " << element_size_bytes << std::endl;
      std::cout << "num_elements " << num_elements << std::endl;
      auto size = num_elements * element_size_bytes;
      size_all += size;

      std::cout << "k " << k << std::endl;
      std::cout << "v " << v << std::endl;
  }
  std::cout << "func->buffer_map end " << std::endl;

  // this->VisitStmt(func->body);

  // size * dtype.bytes()
  return size_all;
  // return this->max_size;
}

// template <typename T>
// size_t InOutCalculator<T>::GetByteAlignedSize(Integer non_aligned_size) {
//   return non_aligned_size.defined()
//              ? ((non_aligned_size.IntValue() + byte_alignment - 1) / byte_alignment) *
//                    byte_alignment
//              : 0;
// }

template <typename T>
void InOutCalculator<T>::VisitStmt_(const T* op) {

  std::cout << "InOutCalculator::VisitStmt_" << std::endl;

  const DataType& dtype = op->dtype;
  const Array<PrimExpr>& extents = op->extents;
  std::cout << "op->buffer_var " << op->buffer_var << std::endl;
  // op->buffer_var
  size_t num_elements = 1;
  Integer res = Integer();
  for (const auto& ext : extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    } else {
      // We can't statically calculate workspace for dynamic shapes
      break;
    }
  }
  size_t element_size_bytes = dtype.bytes();

  res = Integer(num_elements * element_size_bytes);

  auto size = res.defined() ? res.IntValue() : 0;

  // auto size = GetByteAlignedSize(usmp::CalculateExtentsSize(op));
  current_size += size;
  if (current_size > max_size) {
    max_size = current_size;
  }
  StmtExprVisitor::VisitStmt(op->body);
  current_size -= size;
}

size_t CalculateIntOutBytes(const PrimFunc& func) {
  std::cout << "CalculateIntOutBytes" << std::endl;
  // InOutCalculator<AllocateConstNode> wc;
  // InOutCalculator<AttrStmtNode> wc;
  InOutCalculator<AllocateNode> wc;
  // BufferStoreNode
  // wc.byte_alignment = byte_alignment->value;
  return wc(func);
}

// size_t CalculateWorkspaceBytes(const PrimFunc& func, const Integer& byte_alignment) {
//   InOutCalculator<AllocateNode> wc;
//   wc.byte_alignment = byte_alignment->value;
//   return wc(func);
// }

TVM_REGISTER_GLOBAL("tir.analysis.calculate_intout_bytes")
    .set_body_typed([](PrimFunc func) {
      return static_cast<int>(CalculateIntOutBytes(func));
    });

// TVM_REGISTER_GLOBAL("tir.analysis.calculate_workspace_bytes")
//     .set_body_typed([](PrimFunc func, Integer workspace_byte_alignment) {
//       return static_cast<int>(CalculateWorkspaceBytes(func, workspace_byte_alignment));
//     });

}  // namespace tir
}  // namespace tvm
