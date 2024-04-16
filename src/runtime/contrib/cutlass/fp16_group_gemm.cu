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

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <optional>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "group_gemm_runner.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <>
struct KernelTraits<cutlass::half_t> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _256, _64>;  // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;    // Shape of the threadblocks in a cluster
};

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

namespace fastertransformer {

template <typename T, typename WeightType>
void moe_gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases,
                       T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n,
                       int64_t gemm_k, int num_experts, std::optional<std::string> activation,
                       cudaStream_t stream);
}

__global__ void prepare_group_gemm_lora_arguments(
    cutlass::half_t** ptr_A, cutlass::half_t** ptr_B, cutlass::half_t** ptr_D,
    cutlass::gemm::GemmCoord* problem_sizes, int64_t* stride_A, int64_t* stride_B, int64_t* stride_D,
    cutlass::half_t* x, cutlass::half_t* weight, cutlass::half_t* out,
    int64_t* indptr, int32_t* ranks, int32_t* active_slots, int64_t n, int64_t k, int64_t num_groups) {
  int group_id = threadIdx.x;
  if (group_id >= num_groups) return;
  int prev_rows = group_id == 0 ? 0 : indptr[group_id - 1];
  ptr_A[group_id] = x + prev_rows * k;
  ptr_B[group_id] = weight + active_slots[group_id] * k * n;
  ptr_D[group_id] = out + prev_rows * n;

  if (n > k) {
    // LoRA B, reduce only over the valid rank range
    problem_sizes[group_id] = {static_cast<int>(indptr[group_id] - prev_rows), static_cast<int>(n),
                               ranks[group_id]};
  } else {
    problem_sizes[group_id] = {static_cast<int>(indptr[group_id] - prev_rows), static_cast<int>(n),
                               static_cast<int>(k)};
  }

  stride_A[group_id] = k;
  stride_B[group_id] = k;
  stride_D[group_id] = n;
}

template <typename TileShape, typename WarpShape, typename InstShape, int EpilogueVecSize=8>
void cutlass_group_gemm_lora_sm80(cutlass::half_t* x, cutlass::half_t* weight, int64_t* indptr, int32_t* ranks,
				  int32_t* active_slots, uint8_t* workspace,
				  int64_t workspace_size, int n, int k, int num_groups,
				  float alpha, float beta, cutlass::half_t* out, cudaStream_t stream) {
  std::ptrdiff_t offset = 0;
  cutlass::half_t** ptr_A = reinterpret_cast<cutlass::half_t**>(workspace + offset);
  offset += aligned(sizeof(cutlass::half_t*) * num_groups);
  cutlass::half_t** ptr_B = reinterpret_cast<cutlass::half_t**>(workspace + offset);
  offset += aligned(sizeof(cutlass::half_t*) * num_groups);
  cutlass::half_t** ptr_D = reinterpret_cast<cutlass::half_t**>(workspace + offset);
  offset += aligned(sizeof(cutlass::half_t*) * num_groups);
  auto* problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace + offset);
  offset += aligned(sizeof(cutlass::gemm::GemmCoord) * num_groups);
  auto* stride_A = reinterpret_cast<int64_t*>(workspace + offset);
  offset += aligned(sizeof(int64_t) * num_groups);
  auto* stride_B = reinterpret_cast<int64_t*>(workspace + offset);
  offset += aligned(sizeof(int64_t) * num_groups);
  auto* stride_D = reinterpret_cast<int64_t*>(workspace + offset);
  offset += aligned(sizeof(int64_t) * num_groups);
  prepare_group_gemm_lora_arguments<<<1, num_groups, 0, stream>>>(ptr_A, ptr_B, ptr_D, problem_sizes,
                                                                  stride_A, stride_B, stride_D, x,
                                                                  weight, out, indptr, ranks, active_slots, n, k, num_groups);
  offset = aligned(offset, 256);

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      cutlass::half_t,                                      // Element A
      cutlass::layout::RowMajor,                      // Layout A
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity A
      cutlass::half_t,                                      // Element B
      cutlass::layout::ColumnMajor,                      // Layout B
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity B
      cutlass::half_t,                                      // Element C&D
      cutlass::layout::RowMajor,                      // Layout C&D
      float,                                          // Element Accumulator
      cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
      cutlass::arch::Sm80,                            // Architecture
      TileShape,
      WarpShape,
      InstShape,
      LinearCombination<cutlass::half_t, EpilogueVecSize, float, float>,  // Epilogue
      GemmIdentityThreadblockSwizzle<1>,              // Swizzling Operator
      2                                               // Stages
      >::GemmKernel;

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(alpha, beta);

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(problem_sizes, num_groups, 512,
                                       epilogue_op, ptr_A, ptr_B, ptr_D,
                                       ptr_D, stride_A, stride_B, stride_D, stride_D);

  GemmGrouped gemm;
  auto status = gemm.initialize(args, workspace + offset, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "Grouped gemm initialization failed: %s\n",
            cutlassGetStatusString(status));
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "Grouped gemm run failed: %s\n",
            cutlassGetStatusString(status));
  }
}

namespace tvm {
namespace runtime {

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_group_gemm_sm90(NDArray x, NDArray weight, NDArray indptr,
                                 NDArray workspace, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = indptr->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  if (x->shape[0] <= 4 && k % 8 == 0) {
    fastertransformer::moe_gemm_bias_act<half, half>(
        reinterpret_cast<half*>(x->data), reinterpret_cast<half*>(weight->data), nullptr, nullptr,
        reinterpret_cast<half*>(out->data), reinterpret_cast<int64_t*>(indptr->data), x->shape[0],
        n, k, num_groups, std::nullopt, stream);
    return;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  using TileShape = Shape<_128, _256, _64>;
  cutlass_group_gemm<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
		                static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
                                workspace->shape[0], n, k, num_groups, alpha, beta,
                                static_cast<ElementC*>(out->data), stream);
}

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_group_gemm_sm90_scale(NDArray x, NDArray weight, NDArray indptr, NDArray workspace,
                                       NDArray alpha, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(alpha->dtype.code, kDLFloat);
  CHECK_EQ(alpha->dtype.bits, 32);
  CHECK_EQ(out->ndim, 2);
  int num_groups = weight->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  using TileShape = Shape<_128, _256, _64>;
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
  cutlass_group_gemm<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
				static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
				workspace->shape[0], n, k, num_groups, static_cast<float*>(alpha->data),
				static_cast<float*>(nullptr), static_cast<ElementC*>(out->data), stream);
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_fp16_sm90")
    .set_body_typed(tvm_cutlass_group_gemm_sm90<cutlass::half_t, cutlass::half_t, cutlass::half_t>);

TVM_REGISTER_GLOBAL("cutlass.group_gemm_scale_fp16_sm90")
    .set_body_typed(
        tvm_cutlass_group_gemm_sm90_scale<cutlass::half_t, cutlass::half_t, cutlass::half_t>);

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_group_gemm_lora_sm90(NDArray x, NDArray weight, NDArray indptr,
				      NDArray ranks, NDArray active_slots,
				      NDArray workspace, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = indptr->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  float alpha = 1.0f;
  float beta = 0.0f;

  int32_t* active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  int32_t* ranks_ptr = static_cast<int32_t*>(ranks->data);

  // Small N specialization for LoRA
  if (n <= 16) {
    using TileShape = Shape<_128, _16, _64>;
    cutlass_group_gemm_lora<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
     		  	          static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
                                  workspace->shape[0], n, k, num_groups, alpha, beta,
                                  static_cast<ElementC*>(out->data), stream);
  } else if (n <= 32) {
    using TileShape = Shape<_128, _32, _64>;
    cutlass_group_gemm_lora<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
                                  static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
                                  workspace->shape[0], n, k, num_groups, alpha, beta,
                                  static_cast<ElementC*>(out->data), stream);
  } else if (n <= 64) {
    using TileShape = Shape<_128, _64, _64>;
    cutlass_group_gemm_lora<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
				  static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
                                  workspace->shape[0], n, k, num_groups, alpha, beta,
                                  static_cast<ElementC*>(out->data), stream);
  } else {
    using TileShape = Shape<_128, _256, _64>;
    cutlass_group_gemm_lora<TileShape>(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
				  static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
                                  workspace->shape[0], n, k, num_groups, alpha, beta,
                                  static_cast<ElementC*>(out->data), stream);
  }
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_lora_fp16_sm90")
    .set_body_typed(tvm_cutlass_group_gemm_lora_sm90<cutlass::half_t, cutlass::half_t, cutlass::half_t>);

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

void tvm_cutlass_group_gemm_sm80(NDArray x, NDArray weight, NDArray indptr, NDArray out) {
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = weight->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  fastertransformer::moe_gemm_bias_act<half, half>(
      reinterpret_cast<half*>(x->data), reinterpret_cast<half*>(weight->data), nullptr, nullptr,
      reinterpret_cast<half*>(out->data),
      reinterpret_cast<int64_t*>(indptr->data), x->shape[0], n, k, num_groups,
      std::nullopt, stream);
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_fp16_sm80")
    .set_body_typed(tvm_cutlass_group_gemm_sm80);

void tvm_cutlass_group_gemm_lora_sm80(NDArray x, NDArray weight, NDArray indptr,
				      NDArray ranks, NDArray active_slots,
				      NDArray workspace, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = indptr->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  float alpha = 1.0f;
  float beta = 0.0f;

  int32_t* active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  int32_t* ranks_ptr = static_cast<int32_t*>(ranks->data);

  if (n > k) {
    // LoRA B
    auto func = cutlass_group_gemm_lora_sm80<cutlass::gemm::GemmShape<32, 128, 16>,
                                             cutlass::gemm::GemmShape<32, 64, 16>,
                                             cutlass::gemm::GemmShape<16, 8, 8>>;
    func(static_cast<cutlass::half_t*>(x->data), static_cast<cutlass::half_t*>(weight->data),
         static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
         workspace->shape[0], n, k, num_groups, alpha, beta,
         static_cast<cutlass::half_t*>(out->data), stream);
  } else {
    // LoRA A
    auto func = cutlass_group_gemm_lora_sm80<cutlass::gemm::GemmShape<16, 64, 64>,
                                             cutlass::gemm::GemmShape<16, 16, 64>,
                                             cutlass::gemm::GemmShape<16, 8, 16>,
                                             4>;
    func(static_cast<cutlass::half_t*>(x->data), static_cast<cutlass::half_t*>(weight->data),
         static_cast<int64_t*>(indptr->data), ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
         workspace->shape[0], n, k, num_groups, alpha, beta,
         static_cast<cutlass::half_t*>(out->data), stream);
  }

}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_lora_fp16_sm80")
    .set_body_typed(tvm_cutlass_group_gemm_lora_sm80);

}  // namespace runtime
}  // namespace tvm
