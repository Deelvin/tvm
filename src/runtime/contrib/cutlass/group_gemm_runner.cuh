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

#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>
#include <vector>

#include "../../cuda/cuda_common.h"

// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                      \
  {                                                                \
    cutlass::Status error = status;                                \
    CHECK(error == cutlass::Status::kSuccess)                      \
        << "Got cutlass error: " << cutlassGetStatusString(error); \
  }

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

inline size_t aligned(size_t value, size_t alignment = 16) {
  return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
struct KernelTraits;

template <typename ElementA, typename ElementB, typename ElementC, typename StrideA,
          typename StrideB, typename StrideC, typename ProblemShape>
struct GroupedGemmArguments {
  GroupedGemmArguments(uint8_t* workspace_ptr, int num_groups, int ws_size) :
    workspace(workspace_ptr),
    workspace_size(ws_size){
    std::ptrdiff_t offset = 0;
    ptr_A = reinterpret_cast<const ElementA**>(workspace + offset);
    offset += aligned(sizeof(ElementA*) * num_groups);
    ptr_B = reinterpret_cast<const ElementB**>(workspace + offset);
    offset += aligned(sizeof(ElementB*) * num_groups);
    ptr_D = reinterpret_cast<ElementC**>(workspace + offset);
    offset += aligned(sizeof(ElementC*) * num_groups);
    problem_sizes = reinterpret_cast<ProblemShape*>(workspace + offset);
    offset += aligned(sizeof(ProblemShape) * num_groups);
    stride_A = reinterpret_cast<StrideA*>(workspace + offset);
    offset += aligned(sizeof(StrideA) * num_groups);
    stride_B = reinterpret_cast<StrideB*>(workspace + offset);
    offset += aligned(sizeof(StrideB) * num_groups);
    stride_D = reinterpret_cast<StrideC*>(workspace + offset);
    offset += aligned(sizeof(StrideC) * num_groups);
    offset = aligned(offset, 256);
    workspace += offset;
    workspace_size -= offset;
  }


  uint8_t* workspace;
  int workspace_size;
  const ElementA** ptr_A;
  const ElementB** ptr_B;
  ElementC** ptr_D;
  StrideA* stride_A;
  StrideB* stride_B;
  StrideC* stride_D;
  ProblemShape* problem_sizes;
};

template <typename ElementA, typename ElementB, typename ElementC, typename TileShape,
          typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::ColumnMajor,
          typename LayoutC = cutlass::layout::RowMajor>
struct CutlassGroupGemmRunner {
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of elements
                                                    // (up to 16 bytes)

  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of elements
                                                    // (up to 16 bytes)

  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements
                                                    // (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ScaleType = std::variant<ElementAccumulator, const ElementAccumulator*>;
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using ClusterShape = typename KernelTraits<ElementA>::ClusterShape;
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
  using KernelSchedule = typename KernelTraits<ElementA>::KernelSchedule;     // Kernel to launch
  using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;  // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::UnderlyingStrideA;
  using StrideB = typename Gemm::GemmKernel::UnderlyingStrideB;
  using StrideC = typename Gemm::GemmKernel::UnderlyingStrideC;
  using StrideD = typename Gemm::GemmKernel::UnderlyingStrideD;

  using GroupedGemmArgumentsType = GroupedGemmArguments<ElementA, ElementB, ElementC, StrideA, StrideB, StrideC, typename ProblemShape::UnderlyingProblemShape>;

  void run_group_gemm(GroupedGemmArgumentsType arg, int num_groups, ScaleType alpha,
                      ScaleType beta, cudaStream_t stream) {
    typename Gemm::EpilogueOutputOp::Params epilogue_params = [&]() {
      ICHECK(alpha.index() == beta.index()) << "alpha and beta must have the same type";
      if (std::holds_alternative<ElementAccumulator>(alpha)) {
        return typename Gemm::EpilogueOutputOp::Params{std::get<ElementAccumulator>(alpha),
                                                       std::get<ElementAccumulator>(beta)};
      } else if (std::holds_alternative<const ElementAccumulator*>(alpha)) {
        return typename Gemm::EpilogueOutputOp::Params{std::get<const ElementAccumulator*>(alpha),
                                                       std::get<const ElementAccumulator*>(beta)};
      } else {
        LOG(FATAL) << "Unsupported alpha and beta type";
        throw;
      }
    }();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                       {num_groups, arg.problem_sizes, nullptr},
                                       {arg.ptr_A, arg.stride_A, arg.ptr_B, arg.stride_B},
			               {epilogue_params, const_cast<const ElementC**>(arg.ptr_D), arg.stride_D, arg.ptr_D, arg.stride_D},
                                       hw_info};
    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CHECK_GE(arg.workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, arg.workspace, stream));
    CUTLASS_CHECK(gemm_op.run(stream));
  }

};

template <typename GroupedGemmArgumentsType, typename ElementA, typename ElementB, typename ElementC>
__global__ void prepare_group_gemm_arguments(GroupedGemmArgumentsType arg, const ElementA* x, const ElementB* weight, ElementC* out,
    int64_t* indptr, int64_t n, int64_t k, int64_t num_groups) {
  int group_id = threadIdx.x;
  if (group_id >= num_groups) return;
  int prev_rows = group_id == 0 ? 0 : indptr[group_id - 1];
  arg.ptr_A[group_id] = x + prev_rows * k;
  arg.ptr_B[group_id] = weight + group_id * k * n;
  arg.ptr_D[group_id] = out + prev_rows * n;

  arg.problem_sizes[group_id] = {static_cast<int>(indptr[group_id] - prev_rows), static_cast<int>(n),
                                 static_cast<int>(k)};

  arg.stride_A[group_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  arg.stride_B[group_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  arg.stride_D[group_id] = cute::make_stride(n, Int<1>{}, int64_t{0});
}

template <typename TileShape, typename ElementA, typename ElementB, typename ElementC>
void cutlass_group_gemm(ElementA* x, ElementB* weight, int64_t* indptr, uint8_t* workspace,
                        int64_t workspace_size, int64_t n, int64_t k, int64_t num_groups,
                        std::variant<float, const float*> alpha,
                        std::variant<float, const float*> beta, ElementC* out,
                        cudaStream_t stream) {
  using Runner = CutlassGroupGemmRunner<ElementA, ElementB, ElementC, TileShape>;
  typename Runner::GroupedGemmArgumentsType arg(workspace, num_groups, workspace_size);

  prepare_group_gemm_arguments<<<1, num_groups, 0, stream>>>(arg, x, weight, out, indptr, n, k, num_groups);
  Runner runner;
  runner.run_group_gemm(arg, num_groups, alpha, beta, stream);
}

template <typename GroupedGemmArgumentsType, typename ElementA, typename ElementB, typename ElementC>
__global__ void prepare_group_gemm_lora_arguments(GroupedGemmArgumentsType arg,
    const ElementA* x, const ElementB* weight, ElementC* out,
    int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots, int64_t n, int64_t k, int64_t num_groups) {
  int group_id = threadIdx.x;
  if (group_id >= num_groups) return;
  int prev_rows = indices_counts_ex_scan[group_id];
  arg.ptr_A[group_id] = x + prev_rows * k;
  arg.ptr_B[group_id] = weight + active_slots[group_id] * k * n;
  arg.ptr_D[group_id] = out + prev_rows * n;

  if (n > k) {
    // LoRA B, reduce only over the valid rank range
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[group_id + 1] - prev_rows), static_cast<int>(n),
                               ranks[group_id]};
  } else {
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[group_id + 1] - prev_rows), static_cast<int>(n),
                               static_cast<int>(k)};
  }

  arg.stride_A[group_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  arg.stride_B[group_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  arg.stride_D[group_id] = cute::make_stride(n, Int<1>{}, int64_t{0});
}

template <typename TileShape, typename ElementA, typename ElementB, typename ElementC>
void cutlass_group_gemm_lora(ElementA* x, ElementB* weight, int64_t* indices_counts_ex_scan, int32_t* ranks,
			     int32_t* active_slots, uint8_t* workspace,
                             int64_t workspace_size, int64_t n, int64_t k, int64_t num_groups,
                             std::variant<float, const float*> alpha,
                             std::variant<float, const float*> beta, ElementC* out,
                             cudaStream_t stream) {
  using Runner = CutlassGroupGemmRunner<ElementA, ElementB, ElementC, TileShape>;
  typename Runner::GroupedGemmArgumentsType arg(workspace, num_groups, workspace_size);

  prepare_group_gemm_lora_arguments<<<1, num_groups, 0, stream>>>(arg, x, weight, out, indices_counts_ex_scan,
								  ranks, active_slots, n, k, num_groups);
  Runner runner;
  runner.run_group_gemm(arg, num_groups, alpha, beta, stream);
}

template <typename GroupedGemmArgumentsType, typename ElementA, typename ElementB, typename ElementC>
__global__ void prepare_nested_group_gemm_lora_B_arguments(GroupedGemmArgumentsType arg,
    const ElementA* lora_A_out, const ElementB* lora_B1, const ElementB* lora_B2,
    ElementC* out, int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots,
    int64_t n1, int64_t n2, int64_t padded_rank) {
  int outer_id = threadIdx.x;  // combined B weights
  int inner_id = threadIdx.y;  // lora
  int group_id = outer_id * blockDim.y + inner_id;
  int prev_rows = indices_counts_ex_scan[inner_id];
  arg.ptr_A[group_id] = lora_A_out + outer_id * padded_rank + prev_rows * padded_rank * 2;
  arg.ptr_D[group_id] = out + outer_id * n1 + prev_rows * (n1 + n2);

  if (outer_id == 0) {
    arg.ptr_B[group_id] = lora_B1 + active_slots[inner_id] * n1 * padded_rank;
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[inner_id + 1] - prev_rows), static_cast<int>(n1),
	       		           static_cast<int>(ranks[inner_id])};
  } else {
    arg.ptr_B[group_id] = lora_B2 + active_slots[inner_id] * n2 * padded_rank;
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[inner_id + 1] - prev_rows), static_cast<int>(n2),
			           static_cast<int>(ranks[inner_id])};
  }

  arg.stride_A[group_id] = cute::make_stride(padded_rank * 2, Int<1>{}, int64_t{0});
  arg.stride_B[group_id] = cute::make_stride(padded_rank, Int<1>{}, int64_t{0});
  arg.stride_D[group_id] = cute::make_stride(n1 + n2, Int<1>{}, int64_t{0});
}

template <typename GroupedGemmArgumentsType, typename ElementA, typename ElementB, typename ElementC>
__global__ void prepare_nested_group_gemm_lora_B_arguments(GroupedGemmArgumentsType arg,
    const ElementA* lora_A_out, const ElementB* lora_B1, const ElementB* lora_B2, const ElementB* lora_B3,
    ElementC* out, int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots,
    int64_t n1, int64_t n2, int64_t n3, int64_t padded_rank) {
  int outer_id = threadIdx.x;  // combined B weights
  int inner_id = threadIdx.y;  // lora
  int group_id = outer_id * blockDim.y + inner_id;
  int prev_rows = indices_counts_ex_scan[inner_id];
  arg.ptr_A[group_id] = lora_A_out + outer_id * padded_rank + prev_rows * padded_rank * 3;

  if (outer_id == 0) {
    arg.ptr_B[group_id] = lora_B1 + active_slots[inner_id] * n1 * padded_rank;
    arg.ptr_D[group_id] = out + prev_rows * (n1 + n2 + n3);
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[inner_id + 1] - prev_rows), static_cast<int>(n1),
			           static_cast<int>(ranks[inner_id])};
  } else if (outer_id == 1) {
    arg.ptr_B[group_id] = lora_B2 + active_slots[inner_id] * n2 * padded_rank;
    arg.ptr_D[group_id] = out + n1 + prev_rows * (n1 + n2 + n3);
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[inner_id + 1] - prev_rows), static_cast<int>(n2),
			           static_cast<int>(ranks[inner_id])};
  } else if (outer_id == 2) {
    arg.ptr_B[group_id] = lora_B3 + active_slots[inner_id] * n3 * padded_rank;
    arg.ptr_D[group_id] = out + n1 + n2 + prev_rows * (n1 + n2 + n3);
    arg.problem_sizes[group_id] = {static_cast<int>(indices_counts_ex_scan[inner_id + 1] - prev_rows), static_cast<int>(n3),
			           static_cast<int>(ranks[inner_id])};
  }

  arg.stride_A[group_id] = cute::make_stride(padded_rank * 3, Int<1>{}, int64_t{0});
  arg.stride_B[group_id] = cute::make_stride(padded_rank, Int<1>{}, int64_t{0});
  arg.stride_D[group_id] = cute::make_stride(n1 + n2 + n3, Int<1>{}, int64_t{0});
}

template <typename TileShape, typename ElementA, typename ElementB, typename ElementC>
void cutlass_nested_group_gemm_lora_B_fp16_sm90(ElementA* lora_A_out, std::vector<ElementB*> lora_B_weights, int64_t* indices_counts_ex_scan,
						int32_t* ranks, int32_t* active_slots, int64_t num_loras, int64_t padded_rank, float beta,
						int64_t workspace_size, uint8_t* workspace, const std::vector<int64_t>& out_feature_sizes,
						ElementC* out) {
  using Runner = CutlassGroupGemmRunner<ElementA, ElementB, ElementC, TileShape>;

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  auto num_combined = lora_B_weights.size();
  auto num_groups = num_combined * num_loras;

  typename Runner::GroupedGemmArgumentsType arg(workspace, num_groups, workspace_size);

  dim3 block(num_combined, num_loras);

  if (num_combined == 2) {
    auto n1 = out_feature_sizes[0];
    auto n2 = out_feature_sizes[1];
    prepare_nested_group_gemm_lora_B_arguments<<<1, block, 0, stream>>>(arg, lora_A_out,
								        lora_B_weights[0], lora_B_weights[1],
									out, indices_counts_ex_scan,
									ranks, active_slots,
									n1, n2, padded_rank);

  } else if (num_combined == 3) {
    auto n1 = out_feature_sizes[0];
    auto n2 = out_feature_sizes[1];
    auto n3 = out_feature_sizes[2];
    prepare_nested_group_gemm_lora_B_arguments<<<1, block, 0, stream>>>(arg, lora_A_out,
									lora_B_weights[0], lora_B_weights[1], lora_B_weights[2],
									out, indices_counts_ex_scan,
									ranks, active_slots,
									n1, n2, n3, padded_rank);
  } else {
    LOG(FATAL) << "Unsupported number of outer grouped gemm size: " << num_combined;
  }

  Runner runner;
  runner.run_group_gemm(arg, num_groups, 1.0, beta, stream);
}
