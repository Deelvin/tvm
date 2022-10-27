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

 private:
  void VisitStmt_(const T* op) override;
  size_t current_size = 0;
  size_t max_size = 0;
};





template <typename T>
size_t InOutCalculator<T>::operator()(const PrimFunc& func) {
  this->VisitStmt(func->body);
  return this->max_size;
}

// std::string GetStorageScope(const Var& var) {
//   auto* ptr = var->type_annotation.as<PointerTypeNode>();
//   ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
//   return ptr->storage_scope;
// }

template <typename T>
void InOutCalculator<T>::VisitStmt_(const T* op) {
  auto size = op->ConstantAllocationSize();
  std::cout << "ICE InOutCalculator::ConstantAllocationSize " << size << std::endl << std::flush;
  // std::string storage_scope = GetStorageScope(op->buffer_var);
  // std::cout << "ICE InOutCalculator::scope " << storage_scope << std::endl << std::flush;
  // static_cast<AllocateNode*>(op)->extents
  current_size += size;
  if (current_size > max_size) {
    max_size = current_size;
  }
  StmtExprVisitor::VisitStmt(op->body);
  current_size -= size;
}

// static Integer CalculateExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents) {
//   size_t element_size_bytes = dtype.bytes();
//   size_t num_elements = 1;
//   for (const auto& ext : extents) {
//     if (ext->IsInstance<IntImmNode>()) {
//       num_elements *= Downcast<IntImm>(ext)->value;
//     } else {
//       // We can't statically calculate workspace for dynamic shapes
//       return Integer();
//     }
//   }
//   return Integer(num_elements * element_size_bytes);
// }
// template <typename T>
// size_t InOutCalculator<T>::operator()(const PrimFunc& func) {
//   // std::cout << "InOutCalculator::operator" << std::endl << std::flush;
//   // std::cout << "func->body\n" << func->body << std::endl << std::flush;

//   // size_t element_size_bytes = dtype.bytes();
//   // size_t size = 0;

// // ConstantAllocationSize
//   for (size_t i = 0; i < func->params.size(); i++) {
//     // func->params[i]
//     std::cout << "func->params[i] " << i << " " <<  func->params[i] << std::endl << std::flush;
//     std::cout << "func->params[i]->dtype " << func->params[i]->dtype.bytes() << std::endl << std::flush;
//     // CHECK(func->params[i]->dtype.is_handle()) << "ValueError: Parameters of the description of the "
//     //                       "tensor intrinsic should be handle only.";
//   }
//   std::cout << "func->body" << std::endl << std::flush;
//   std::cout << "func->buffer_map.size() " << func->buffer_map.size() << std::endl << std::flush;



//   size_t size_all = 0;
//   for (auto [k,v] : func->buffer_map) {
//       // v->
    
//       Integer num_elements;
//       // size_t num_elements = 1;
//       Integer res = Integer();


//       num_elements = CalculateExtentsSize(v->dtype, v->shape);
//       // for (const auto& ext : v->shape) {
//       //   // if (true) {
//       //   if (ext->IsInstance<IntImmNode>()) {
//       //     num_elements *= Downcast<IntImm>(ext)->value;
//       //   } else {
//       //     // We can't statically calculate workspace for dynamic shapes
//       //     num_elements = 0;
//       //     break;
//       //   }
//       // }
//       size_t element_size_bytes = v->dtype.bytes();
//       // std::cout << "element_size_bytes " << element_size_bytes << std::endl << std::flush;
//       // std::cout << "num_elements " << num_elements.IntValue() / element_size_bytes << std::endl << std::flush;
      
//       if (!num_elements.defined()) {
//       // if (num_elements == 0) {
//         size_all = 0;
//         break;
//       }
//       // size_all = num_elements.IntValue();
      
//       // auto size = num_elements * element_size_bytes;
//       size_all += num_elements.IntValue();
//       // size_all += size;

//       std::cout << "k " << k << std::endl << std::flush;
//       std::cout << "v " << v << std::endl << std::flush;
//       std::cout << "v.scope() " << v.scope() << std::endl << std::flush;
//   }
//   std::cout << "func->buffer_map end" << std::endl << std::flush;

//   // this->VisitStmt(func->body);

//   // size * dtype.bytes()
//   return size_all;
//   // return this->max_size;
// }

// template <typename T>
// size_t InOutCalculator<T>::GetByteAlignedSize(Integer non_aligned_size) {
//   return non_aligned_size.defined()
//              ? ((non_aligned_size.IntValue() + byte_alignment - 1) / byte_alignment) *
//                    byte_alignment
//              : 0;
// }

// template <typename T>
// void InOutCalculator<T>::VisitStmt_(const T* op) {

//   // std::cout << "InOutCalculator::VisitStmt_" << std::flush <<  std::endl;

//   const DataType& dtype = op->dtype;
//   const Array<PrimExpr>& extents = op->extents;
//   // std::cout << "op->buffer_var " << op->buffer_var << std::flush <<  std::endl;
//   // op->buffer_var
//   size_t num_elements = 1;
//   Integer res = Integer();
//   for (const auto& ext : extents) {
//     if (ext->IsInstance<IntImmNode>()) {
//       num_elements *= Downcast<IntImm>(ext)->value;
//     } else {
//       // We can't statically calculate workspace for dynamic shapes
//       break;
//     }
//   }
//   size_t element_size_bytes = dtype.bytes();

//   res = Integer(num_elements * element_size_bytes);

//   auto size = res.defined() ? res.IntValue() : 0;

//   // auto size = GetByteAlignedSize(usmp::CalculateExtentsSize(op));
//   current_size += size;
//   if (current_size > max_size) {
//     max_size = current_size;
//   }
//   StmtExprVisitor::VisitStmt(op->body);
//   current_size -= size;
// }

size_t CalculateIntOutTensorsBytes(const PrimFunc& func) {
  // InOutCalculator<AllocateConstNode> wc;
  // InOutCalculator<AttrStmtNode> wc;
  InOutCalculator<AllocateNode> wc;
  // BufferStoreNode
  // wc.byte_alignment = byte_alignment->value;
  auto size = wc(func);
  // std::cout << "CalculateIntOutTensorsBytes size:" << size <<  std::endl << std::flush;
  return size;
}

// size_t CalculateWorkspaceBytes(const PrimFunc& func, const Integer& byte_alignment) {
//   InOutCalculator<AllocateNode> wc;
//   wc.byte_alignment = byte_alignment->value;
//   return wc(func);
// }

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
        std::cout << "ICE VerifySRAMLimit" << kv.first << " size: " << size << std::endl << std::flush;
        
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
