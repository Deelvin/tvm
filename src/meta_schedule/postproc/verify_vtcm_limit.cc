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
namespace tir {

// class ThreadExtentChecker : private StmtVisitor {
//  public:
//   static bool Check(const Stmt& stmt, int thread_warp_size) {
//     try {
//       ICHECK(thread_warp_size > 0);
//       ThreadExtentChecker checker(thread_warp_size);
//       checker.VisitStmt(stmt);
//       return true;
//     } catch (const dmlc::Error& e) {
//       return false;
//     }
//   }

//  private:
//   explicit ThreadExtentChecker(int thread_warp_size) : thread_warp_size_(thread_warp_size) {}

//   void VisitStmt_(const ForNode* loop) {
//     runtime::ThreadScope thread_scope = GetThreadScope(loop);
//     if (IsThreadIdx(thread_scope)) {
//       if (const int64_t* p_ext = GetLoopIntExtent(loop)) {
//         int64_t ext = *p_ext;
//         if (thread_scope.dim_index == 0) {
//           std::swap(thread_idx_x, ext);
//           StmtVisitor::VisitStmt_(loop);
//           std::swap(thread_idx_x, ext);
//         } else if (thread_scope.dim_index == 1) {
//           std::swap(thread_idx_y, ext);
//           StmtVisitor::VisitStmt_(loop);
//           std::swap(thread_idx_y, ext);
//         } else if (thread_scope.dim_index == 2) {
//           std::swap(thread_idx_z, ext);
//           StmtVisitor::VisitStmt_(loop);
//           std::swap(thread_idx_z, ext);
//         } else {
//           StmtVisitor::VisitStmt_(loop);
//         }
//         return;
//       } else {
//         throw dmlc::Error("Dynamic thread extent");
//       }
//     }
//     StmtVisitor::VisitStmt_(loop);
//   }

//   void VisitStmt_(const BlockNode* block) {
//     int old_thread_idx_x = thread_idx_x;
//     if (block->annotations.count(attr::warp_execution)) {
//       thread_idx_x = thread_warp_size_;
//     }
//     if (Optional<Integer> low_inclusive =
//             GetAnn<Integer>(block, attr::meta_schedule_thread_extent_low_inclusive)) {
//       if (Optional<Integer> high_inclusive =
//               GetAnn<Integer>(block, attr::meta_schedule_thread_extent_high_inclusive)) {
//         int64_t low = low_inclusive.value()->value;
//         int64_t high = high_inclusive.value()->value;
//         int64_t thread_extent_product = thread_idx_x * thread_idx_y * thread_idx_z;
//         if (!(low <= thread_extent_product && thread_extent_product <= high)) {
//           throw dmlc::Error("Thread extent");
//         }
//       }
//     }
//     StmtVisitor::VisitStmt_(block);
//     thread_idx_x = old_thread_idx_x;
//   }

//   int64_t thread_idx_x = 1;
//   int64_t thread_idx_y = 1;
//   int64_t thread_idx_z = 1;
//   int thread_warp_size_ = -1;
// };

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief Extract attribute from a target. */
Integer Extract__(const Target& target, const char* name) {
  ICHECK(target.defined());
  if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
    return v.value();
  }
  LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
  throw;
}

/*! \brief Verify the correctness of the generated GPU code. */
class VerifyVTCMLimitNode : public PostprocNode { // ICE
 public:
  // Map<String, PrimExpr> target_constraints_{nullptr};
  // int thread_warp_size_ = -1;
  int vtcm_capacity = -1;

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();
    // this->target_constraints_ = Map<String, PrimExpr>{
    //     {"max_shared_memory_per_block", Extract(target, "max_shared_memory_per_block")},
    //     {"max_threads_per_block", Extract(target, "max_threads_per_block")},
    //     {"max_vthread", Integer(8)},
    //     {"max_vector_bytes", Integer(16)},
    // };
    vtcm_capacity = Extract__(target, "vtcm-capacity").IntValue();
  }

  bool Verify(const IRModule& mod) const {
    int limit = vtcm_capacity;
    for (const auto& kv : mod->functions) {
      if (auto* n = kv.second.as<tir::PrimFuncNode>()) {
        auto func = GetRef<tir::PrimFunc>(n);
        auto sizes = CalculateAllocatedBytes(func);
        static const char vtcm_name[] = "global.vtcm";
        auto search = sizes.find(vtcm_name);
        size_t vtcm_allocated = search != sizes.end() ? sizes[vtcm_name] : 0;

        std::cout << "POST VerifyVTCMLimit" << " global.vtcm memory allocation (allocated: "
            << vtcm_allocated << ", limit: " << limit << ")" << std::endl << std::flush;
        
        if (limit > 0 && vtcm_allocated > limit) {
          std::cout << "POST RuntimeError: The global.vtcm memory allocation limit has been exceeded(allocated: "
            << vtcm_allocated << ", limit: " << limit<< ").\n" << std::endl << std::flush;
          // LOG(FATAL) << "RuntimeError: The global.vtcm memory allocation limit has been exceeded(allocated: "
          //   << vtcm_allocated << ", limit: " << limit<< ").\n" << "In function\n" << func;
          return false;
        }
      }

      // if (const auto* prim_func = kv.second.as<tir::PrimFuncNode>()) {
      //   // if (!tir::VerifyVTCMLimit(GetRef<tir::PrimFunc>(prim_func), this->target_constraints_)) {
      //   if (!tir::VerifyVTCMLimit(GetRef<tir::PrimFunc>(prim_func), this->target_constraints_)) {
      //     return false;
      //   }
      // }
    }
    return true;
  }

  bool Apply(const tir::Schedule& sch) final {
    std::cout << "VerifyVTCMLimitNode::Apply" << std::endl << std::flush;
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
        // if (!tir::ThreadExtentChecker::Check(prim_func->body, thread_warp_size_)) {
        //   return false;
        // }
        IRModule lowered{nullptr};
        try {
          auto pass_list = Array<tvm::transform::Pass>();
          // Phase 1
          // First three passes are not needed in TIR schedule.
          // pass_list.push_back(tir::transform::InjectPrefetch());
          // pass_list.push_back(tir::transform::TextureFlatten());
          // pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
          pass_list.push_back(tir::transform::LowerCrossThreadReduction());
          pass_list.push_back(tir::transform::LowerInitBlock());
          pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
          pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
          pass_list.push_back(tir::transform::UnifyThreadBinding());
          pass_list.push_back(tir::transform::CompactBufferAllocation());
          pass_list.push_back(tir::transform::LowerMatchBuffer());
          pass_list.push_back(tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(tir::transform::LowerOpaqueBlock());
          pass_list.push_back(tir::transform::FlattenBuffer());
          pass_list.push_back(tir::transform::BF16Legalize());
          pass_list.push_back(tir::transform::NarrowDataType(32));
          pass_list.push_back(tir::transform::Simplify());
          // Phase 2
          pass_list.push_back(tir::transform::VectorizeLoop(true));
          pass_list.push_back(tir::transform::InjectVirtualThread());
          pass_list.push_back(tir::transform::InjectDoubleBuffer());
          pass_list.push_back(tir::transform::StorageRewrite());
          pass_list.push_back(tir::transform::MergeDynamicSharedMemoryAllocations());
          // Convert Function to IRModule
          transform::PassContext pass_ctx = transform::PassContext::Current();
          tir::PrimFunc f = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol",
                                     runtime::String(g_var->name_hint));
          bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
          if (noalias) {
            f = WithAttr(std::move(f), "tir.noalias", Bool(true));
          }
          IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod)); // ICE CHECK 
        } catch (const dmlc::Error& e) {
          std::cout << "const dmlc::Error& e" << e.what() << std::endl << std::flush;
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
    // n->target_constraints_ = this->target_constraints_;
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
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyVTCMLimit").set_body_typed(Postproc::VerifyVTCMLimit);

}  // namespace meta_schedule
}  // namespace tvm
