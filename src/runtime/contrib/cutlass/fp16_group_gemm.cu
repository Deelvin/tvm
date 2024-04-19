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

namespace tensorrt_llm
{

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void load(T0* dst, T1* src, size_t offset = 0)
{
    *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<const VecType*>(src) + offset);
}

template<int NPerBlock, int Batch, int BlockSize, bool Inplace>
__device__ void half_batched_gemv_impl(const half* weight, const half* in, half* out,
				       int n, int k_full, int k_reduce, uint8_t* shmem, int n_offset=0)
{
    static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0));
    // using 128 bit global access
    static constexpr int kAccessSize = 128;
    static constexpr int kElemsPerThread = kAccessSize / (sizeof(half) * 8);
    using AccessType = uint4;

    constexpr int WarpSize = 32;
    constexpr int Num = Batch * NPerBlock;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_start_id = bid * NPerBlock;

    // Weight: Column major (n, k)
    weight += (n_start_id - n_offset) * k_full;

    float(*sm)[Num] = reinterpret_cast<float(*)[Num]>(shmem);

    // In order to take advantage of hfma2, we use fp16 for accumulation within threads and fp32 for accumulation
    // between threads.
    half accumulator[Num];
    for (int i = 0; i < Num; ++i)
    {
        accumulator[i] = __float2half_rn(0.f);
    }

    // Iteration in k dimensions
    for (int local_k = tid * kElemsPerThread; local_k < k_reduce; local_k += BlockSize * kElemsPerThread)
    {
        half weights_v[kElemsPerThread * NPerBlock];

        if constexpr (NPerBlock == 1)
        {
            load<AccessType>(weights_v, weight + local_k);
        }
        else
        {
            half weights_vec_k[kElemsPerThread];
#pragma unroll
            for (int x = 0; x < NPerBlock; ++x)
            {
                load<AccessType>(weights_vec_k, weight + x * k_full + local_k);
#pragma unroll
                for (int i = 0; i < kElemsPerThread; ++i)
                {
                    weights_v[i * NPerBlock + x] = weights_vec_k[i];
                }
            }
        }

#pragma unroll
        for (int b = 0; b < Batch; ++b)
        {
            half in_v[kElemsPerThread];
            // load activation elements
            load<AccessType>(in_v, in + b * k_full + local_k);
            // Perform vector inner product and accumulate
            if constexpr (NPerBlock == 1)
            {
                half2 v = __float2half2_rn(0.f);
#pragma unroll
                for (int y = 0; y < kElemsPerThread; y += 2)
                {
                    v = __hfma2(*reinterpret_cast<half2*>(weights_v + y), *reinterpret_cast<half2*>(in_v + y), v);
                }
                accumulator[b] += __hadd(v.x, v.y);
            }
            else
            {
#pragma unroll
                for (int x = 0; x < NPerBlock / 2; ++x)
                {
#pragma unroll
                    for (int y = 0; y < kElemsPerThread; ++y)
                    {
                        *reinterpret_cast<half2*>(accumulator + b * NPerBlock + x * 2)
                            = __hfma2(*reinterpret_cast<half2*>(weights_v + y * NPerBlock + x * 2),
                                __half2half2(in_v[y]), *reinterpret_cast<half2*>(accumulator + b * NPerBlock + x * 2));
                    }
                }
            }
        }
    }
    float reses[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        reses[i] = __half2float(accumulator[i]);
    }

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    __syncwarp();
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            reses[i] += __shfl_xor_sync(~0, reses[i], offset);
        }
    }
    if (tid % WarpSize == 0)
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            sm[tid / WarpSize][i] = reses[i];
        }
    }
    __syncthreads();

    // Each thread is responsible for the accumulation and store to global memory of one element
    for (int i = tid; i < Num; i += BlockSize)
    {
        int nid = i % NPerBlock;
        float v = 0.f;
        for (int j = 0; j < BlockSize / WarpSize; ++j)
        {
            v += sm[j][i];
        }
        int b = i / NPerBlock;

	if constexpr (Inplace) {
	  out[b * n + n_start_id + nid] += __float2half_rn(v);
	} else {
	  out[b * n + n_start_id + nid] = __float2half_rn(v);
	}
    }
}

template <int NPerBlock, int Batch, int BlockSize, bool Inplace>
__global__ void grouped_gemv_lora_f16_kernel(const half* weight, const half* in, half* out, int64_t* indices_counts_ex_scan,
					     int32_t* ranks, int32_t* active_slots, int64_t n, int64_t k, int num_lora)
{
    extern __shared__ uint8_t shmem[];

    const int gid = blockIdx.y;
    int k_reduce = k;

    for (int i = 0; i < num_lora; i++)
    {
        if (indices_counts_ex_scan[i + 1] >= (gid + 1))
        {
            weight += active_slots[i] * n * k;
            in += gid * k;
            out += gid * n;
	    if (n > k) {
	      // LoRA B, reduce only over the unpadded rank
	      k_reduce = ranks[i];
	    }
            break;
        }
    }

    half_batched_gemv_impl<NPerBlock, Batch, BlockSize, Inplace>(weight, in, out, n, k, k_reduce, shmem);
}

void grouped_gemv_lora(const half* A, const half* B, half* C, int64_t* indices_counts_ex_scan,
		       int32_t* ranks, int32_t* active_slots, int64_t num_lora_tokens,
		       int64_t gemm_n, int64_t gemm_k, int num_lora, bool inplace, cudaStream_t stream)
{
    constexpr int Batch = 1;
    constexpr int BlockSize = 128;
    constexpr int NPerBlock = 1;
    dim3 grid(gemm_n / NPerBlock, num_lora_tokens);
    dim3 block(BlockSize);
    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock;

    if (inplace) {
      grouped_gemv_lora_f16_kernel<NPerBlock, Batch, BlockSize, true><<<grid, block, size, stream>>>(
        B, A, C, indices_counts_ex_scan, ranks, active_slots, gemm_n, gemm_k, num_lora);
    } else {
      grouped_gemv_lora_f16_kernel<NPerBlock, Batch, BlockSize, false><<<grid, block, size, stream>>>(
        B, A, C, indices_counts_ex_scan, ranks, active_slots, gemm_n, gemm_k, num_lora);
    }
}

template <int NPerBlock, int Batch, int BlockSize, bool Inplace>
__global__ void nested_grouped_gemv_lora_B_f16_kernel(const half* lora_A_out, const half* lora_B_1, const half* lora_B_2,
						      half* out, int64_t* indices_counts_ex_scan,
						      int32_t* ranks, int32_t* active_slots,
						      int64_t n1, int64_t n2, int64_t padded_rank, int num_lora)
{
    extern __shared__ uint8_t shmem[];

    const int gid = blockIdx.y;
    const int outer_id = blockIdx.x < n1 ? 0 : 1;

    const half* weight = outer_id == 0 ? lora_B_1 : lora_B_2;
    int n = outer_id == 0 ? n1 : n2;
    int n_offset = outer_id == 0 ? 0 : n1;
    int rank = -1;

    for (int i = 0; i < num_lora; i++)
    {
        if (indices_counts_ex_scan[i + 1] >= (gid + 1))
        {
            weight += active_slots[i] * n * padded_rank;
            lora_A_out += gid * padded_rank * 2 + outer_id * padded_rank;
            out += gid * (n1 + n2);
            rank = ranks[i];
            break;
        }
    }

    half_batched_gemv_impl<NPerBlock, Batch, BlockSize, Inplace>(weight, lora_A_out, out, n, padded_rank, rank, shmem, n_offset);
}

template <int NPerBlock, int Batch, int BlockSize, bool Inplace>
__global__ void nested_grouped_gemv_lora_B_f16_kernel(const half* lora_A_out, const half* lora_B_1, const half* lora_B_2, const half* lora_B_3,
						      half* out, int64_t* indices_counts_ex_scan,
						      int32_t* ranks, int32_t* active_slots, int64_t n1, int64_t n2, int64_t n3,
						      int64_t padded_rank, int num_lora)
{
    extern __shared__ uint8_t shmem[];

    const int gid = blockIdx.y;
    const int outer_id = blockIdx.x < n1 ? 0 : (blockIdx.x < n1 + n2 ? 1 : 2);

    const half* weight = outer_id == 0 ? lora_B_1 : (outer_id == 1 ? lora_B_2 : lora_B_3);
    int n = outer_id == 0 ? n1 : (outer_id == 1 ? n2 : n3);
    int n_offset = outer_id == 0 ? 0 : (outer_id == 1 ? n1 : n1 + n2);
    int rank = -1;

    for (int i = 0; i < num_lora; i++)
    {
        if (indices_counts_ex_scan[i + 1] >= (gid + 1))
        {
            weight += active_slots[i] * n * padded_rank;
            lora_A_out += gid * padded_rank * 3 + outer_id * padded_rank;
            out += gid * (n1 + n2 + n3);
            rank = ranks[i];
            break;
        }
    }

    half_batched_gemv_impl<NPerBlock, Batch, BlockSize, Inplace>(weight, lora_A_out, out, n, padded_rank, rank, shmem, n_offset);
}

void nested_grouped_gemv_lora_B(const half* lora_A_out, const std::vector<const half*>& lora_B_weights, half* out,
				int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots, int64_t num_lora_tokens,
				const std::vector<int64_t>& gemm_n, int64_t padded_rank, int num_lora, bool inplace, cudaStream_t stream)
{
    constexpr int Batch = 1;
    constexpr int BlockSize = 128;
    constexpr int NPerBlock = 1;
    dim3 block(BlockSize);
    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock;

    if (lora_B_weights.size() == 2) {
      dim3 grid((gemm_n[0] + gemm_n[1]) / NPerBlock, num_lora_tokens);
      if (inplace) {
	nested_grouped_gemv_lora_B_f16_kernel<NPerBlock, Batch, BlockSize, true><<<grid, block, size, stream>>>(
	  lora_A_out, lora_B_weights[0], lora_B_weights[1], out, indices_counts_ex_scan, ranks, active_slots,
	  gemm_n[0], gemm_n[1], padded_rank, num_lora);
      } else {
	nested_grouped_gemv_lora_B_f16_kernel<NPerBlock, Batch, BlockSize, false><<<grid, block, size, stream>>>(
	  lora_A_out, lora_B_weights[0], lora_B_weights[1], out, indices_counts_ex_scan, ranks, active_slots,
	  gemm_n[0], gemm_n[1], padded_rank, num_lora);
      }
    } else if (lora_B_weights.size() == 3) {
      dim3 grid((gemm_n[0] + gemm_n[1] + gemm_n[2]) / NPerBlock, num_lora_tokens);
      if (inplace) {
	nested_grouped_gemv_lora_B_f16_kernel<NPerBlock, Batch, BlockSize, true><<<grid, block, size, stream>>>(
         lora_A_out, lora_B_weights[0], lora_B_weights[1], lora_B_weights[2], out, indices_counts_ex_scan, ranks, active_slots,
	 gemm_n[0], gemm_n[1], gemm_n[2], padded_rank, num_lora);
      } else {
	nested_grouped_gemv_lora_B_f16_kernel<NPerBlock, Batch, BlockSize, false><<<grid, block, size, stream>>>(
         lora_A_out, lora_B_weights[0], lora_B_weights[1], lora_B_weights[2], out, indices_counts_ex_scan, ranks, active_slots,
	 gemm_n[0], gemm_n[1], gemm_n[2], padded_rank, num_lora);
      }
    } else {
      LOG(FATAL) << "Unsupported number of outer grouped gemm size: " << lora_B_weights.size();;
    }
}

} // namespace tensorrt_llm

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

constexpr int SMALL_BATCH_LORA_A_THRESHOLD = 64;
constexpr int SMALL_BATCH_LORA_B_THRESHOLD = 4;

void tvm_cutlass_group_gemm_lora_sm90_with_beta(NDArray x, NDArray weight, NDArray indices_counts_ex_scan,
						NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
						float beta, NDArray workspace, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;

  auto x_ptr = static_cast<ElementA*>(x->data);
  auto weight_ptr = static_cast<ElementB*>(weight->data);
  auto indices_counts_ex_scan_ptr = static_cast<int64_t*>(indices_counts_ex_scan->data);
  auto active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  auto ranks_ptr = static_cast<int32_t*>(ranks->data);
  auto out_ptr = static_cast<ElementC*>(out->data);

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indices_counts_ex_scan->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);

  int num_groups = active_slots->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  bool is_lora_b = n > k;

  if ((!is_lora_b && num_lora_tokens <= SMALL_BATCH_LORA_A_THRESHOLD) ||
      (is_lora_b && num_lora_tokens <= SMALL_BATCH_LORA_B_THRESHOLD)) {
    bool inplace = beta > 0;
    tensorrt_llm::grouped_gemv_lora(static_cast<half*>(x->data),
				    static_cast<half*>(weight->data),
				    static_cast<half*>(out->data),
				    indices_counts_ex_scan_ptr,
				    ranks_ptr,
				    active_slots_ptr,
				    num_lora_tokens, n, k, num_groups, inplace, stream);
    return;
  }

  float alpha = 1.0f;

  // Small N specialization for LoRA
  if (n <= 16) {
    using TileShape = Shape<_128, _16, _64>;
    cutlass_group_gemm_lora<TileShape>(x_ptr, weight_ptr, indices_counts_ex_scan_ptr,
				       ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
				       workspace->shape[0], n, k, num_groups, alpha, beta,
				       out_ptr, stream);
  } else if (n <= 32) {
    using TileShape = Shape<_128, _32, _64>;
    cutlass_group_gemm_lora<TileShape>(x_ptr, weight_ptr, indices_counts_ex_scan_ptr,
				       ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
				       workspace->shape[0], n, k, num_groups, alpha, beta,
				       out_ptr, stream);
  } else if (n <= 64) {
    using TileShape = Shape<_128, _64, _64>;
    cutlass_group_gemm_lora<TileShape>(x_ptr, weight_ptr, indices_counts_ex_scan_ptr,
				       ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
				       workspace->shape[0], n, k, num_groups, alpha, beta,
				       out_ptr, stream);
  } else {
    using TileShape = Shape<_128, _256, _64>;
    cutlass_group_gemm_lora<TileShape>(x_ptr, weight_ptr, indices_counts_ex_scan_ptr,
				       ranks_ptr, active_slots_ptr, static_cast<uint8_t*>(workspace->data),
				       workspace->shape[0], n, k, num_groups, alpha, beta,
				       out_ptr, stream);
  }
}

void tvm_cutlass_group_gemm_lora_sm90(NDArray x, NDArray weight, NDArray indices_counts_ex_scan,
				      NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
				      NDArray workspace, NDArray out) {
  return tvm_cutlass_group_gemm_lora_sm90_with_beta(x, weight, indices_counts_ex_scan, ranks,
						    active_slots, num_lora_tokens, 0.0f, workspace, out);
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_lora_fp16_sm90")
    .set_body_typed(tvm_cutlass_group_gemm_lora_sm90);

void nested_group_gemm_lora_B_fp16_sm90(NDArray lora_A_out, Array<NDArray> lora_B_weights, NDArray indices_counts_ex_scan,
                                        NDArray ranks, NDArray active_slots, int64_t num_lora_tokens, double beta,
                                        NDArray workspace, NDArray out) {
  auto num_combined = lora_B_weights.size();

  if (num_combined == 1) {
    tvm_cutlass_group_gemm_lora_sm90_with_beta(lora_A_out, lora_B_weights[0], indices_counts_ex_scan,
					       ranks, active_slots, num_lora_tokens, beta, workspace, out);
    return;
  }

  int out_size = 0;
  for (auto arr: lora_B_weights) {
    out_size += arr->shape[1];
  }
  auto num_loras = ranks->shape[0];
  auto padded_rank = lora_B_weights[0]->shape[2];

  CHECK_EQ(lora_A_out->shape[1], padded_rank * num_combined);
  CHECK_EQ(out->shape[1], out_size);
  CHECK_EQ(indices_counts_ex_scan->shape[0], num_loras + 1);

  auto lora_A_out_ptr = static_cast<cutlass::half_t*>(lora_A_out->data);
  auto out_ptr = static_cast<cutlass::half_t*>(out->data);
  auto indices_counts_ex_scan_ptr = static_cast<int64_t*>(indices_counts_ex_scan->data);
  int32_t* active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  int32_t* ranks_ptr = static_cast<int32_t*>(ranks->data);
  auto workspace_ptr = static_cast<uint8_t*>(workspace->data);

  auto get_stream_func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(get_stream_func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*get_stream_func)().operator void*());

  // This does not seem to help
  // if (num_lora_tokens <= SMALL_BATCH_LORA_B_THRESHOLD) {
  //   bool inplace = beta > 0;
  //   std::vector<const half*> lora_B_weights_ptrs;
  //   std::vector<int64_t> out_feature_sizes;
  //   for (auto lora_B: lora_B_weights) {
  //     lora_B_weights_ptrs.push_back(static_cast<const half*>(lora_B->data));
  //     out_feature_sizes.push_back(lora_B->shape[1]);
  //   }

  //   tensorrt_llm::nested_grouped_gemv_lora_B(static_cast<half*>(lora_A_out->data),
  // 					     lora_B_weights_ptrs,
  // 					     static_cast<half*>(out->data),
  // 					     indices_counts_ex_scan_ptr,
  // 					     ranks_ptr, active_slots_ptr,
  // 					     num_lora_tokens, out_feature_sizes,
  // 					     padded_rank, num_loras, inplace, stream);
  //   return;
  // }

  std::vector<cutlass::half_t*> lora_B_weights_ptrs;
  std::vector<int64_t> out_feature_sizes;

  for (auto lora_B: lora_B_weights) {
    lora_B_weights_ptrs.push_back(static_cast<cutlass::half_t*>(lora_B->data));
    out_feature_sizes.push_back(lora_B->shape[1]);
  }

  using TileShape = Shape<_128, _256, _64>;
  auto func = cutlass_nested_group_gemm_lora_B_fp16_sm90<TileShape, cutlass::half_t, cutlass::half_t, cutlass::half_t>;
  func(lora_A_out_ptr, lora_B_weights_ptrs, indices_counts_ex_scan_ptr, ranks_ptr, active_slots_ptr,
       num_loras, padded_rank, beta, workspace->shape[0], workspace_ptr, out_feature_sizes, out_ptr, stream);
}

// Must be called by call_inplace_packed
NDArray nested_group_gemm_lora_B_inplace_fp16_sm90(NDArray lora_A_out, Array<NDArray> lora_B_weights, NDArray indices_counts_ex_scan,
                                                   NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
                                                   NDArray workspace, NDArray base_out) {
  nested_group_gemm_lora_B_fp16_sm90(lora_A_out, lora_B_weights, indices_counts_ex_scan,
				     ranks, active_slots, num_lora_tokens, 1.0f, workspace, base_out);
  return base_out;
}

TVM_REGISTER_GLOBAL("cutlass.nested_group_gemm_lora_B_fp16_sm90")
    .set_body_typed(nested_group_gemm_lora_B_fp16_sm90);

TVM_REGISTER_GLOBAL("cutlass.nested_group_gemm_lora_B_inplace_fp16_sm90")
    .set_body_typed(nested_group_gemm_lora_B_inplace_fp16_sm90);

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


namespace grouped_gemm_sm80 {

struct GroupedGemmArguments {
  GroupedGemmArguments(uint8_t* workspace_ptr, int num_groups) : workspace(workspace_ptr) {
    std::ptrdiff_t offset = 0;
    ptr_A = reinterpret_cast<cutlass::half_t**>(workspace + offset);
    offset += aligned(sizeof(cutlass::half_t*) * num_groups);
    ptr_B = reinterpret_cast<cutlass::half_t**>(workspace + offset);
    offset += aligned(sizeof(cutlass::half_t*) * num_groups);
    ptr_D = reinterpret_cast<cutlass::half_t**>(workspace + offset);
    offset += aligned(sizeof(cutlass::half_t*) * num_groups);
    problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace + offset);
    offset += aligned(sizeof(cutlass::gemm::GemmCoord) * num_groups);
    stride_A = reinterpret_cast<int64_t*>(workspace + offset);
    offset += aligned(sizeof(int64_t) * num_groups);
    stride_B = reinterpret_cast<int64_t*>(workspace + offset);
    offset += aligned(sizeof(int64_t) * num_groups);
    stride_D = reinterpret_cast<int64_t*>(workspace + offset);
    offset += aligned(sizeof(int64_t) * num_groups);
    workspace += aligned(offset, 256);;
  }

  uint8_t* workspace;
  cutlass::half_t** ptr_A;
  cutlass::half_t** ptr_B;
  cutlass::half_t** ptr_D;
  int64_t* stride_A;
  int64_t* stride_B;
  int64_t* stride_D;
  cutlass::gemm::GemmCoord* problem_sizes;
};

__global__ void prepare_group_gemm_lora_arguments(GroupedGemmArguments arg, cutlass::half_t* x, cutlass::half_t* weight,
						  cutlass::half_t* out,  int64_t* indices_counts_ex_scan, int32_t* ranks,
						  int32_t* active_slots, int64_t n, int64_t k, int64_t num_groups) {
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

  arg.stride_A[group_id] = k;
  arg.stride_B[group_id] = k;
  arg.stride_D[group_id] = n;
}

template <typename TileShape, typename WarpShape, typename InstShape, int EpilogueVecSize=8>
void run_group_gemm_sm80(GroupedGemmArguments arg, int num_groups, float alpha, float beta, cudaStream_t stream) {
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
  typename GemmGrouped::Arguments args(arg.problem_sizes, num_groups, 512,
                                       epilogue_op, arg.ptr_A, arg.ptr_B, arg.ptr_D,
                                       arg.ptr_D, arg.stride_A, arg.stride_B, arg.stride_D, arg.stride_D);

  GemmGrouped gemm;
  auto status = gemm.initialize(args, arg.workspace, stream);
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

template <typename TileShape, typename WarpShape, typename InstShape, int EpilogueVecSize=8>
void cutlass_group_gemm_lora_sm80(cutlass::half_t* x, cutlass::half_t* weight, int64_t* indices_counts_ex_scan, int32_t* ranks,
				  int32_t* active_slots, uint8_t* workspace, int n, int k, int num_groups,
				  float alpha, float beta, cutlass::half_t* out, cudaStream_t stream) {
  GroupedGemmArguments arg(workspace, num_groups);
  prepare_group_gemm_lora_arguments<<<1, num_groups, 0, stream>>>(arg, x, weight, out, indices_counts_ex_scan,
								  ranks, active_slots, n, k, num_groups);

  run_group_gemm_sm80<TileShape, WarpShape, InstShape, EpilogueVecSize>(arg, num_groups, alpha, beta, stream);
}

void tvm_cutlass_group_gemm_lora_sm80_with_beta(NDArray x, NDArray weight, NDArray indices_counts_ex_scan,
						NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
						float beta, NDArray workspace, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indices_counts_ex_scan->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = active_slots->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];

  float alpha = 1.0f;

  int64_t* indices_counts_ex_scan_ptr = static_cast<int64_t*>(indices_counts_ex_scan->data);
  int32_t* active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  int32_t* ranks_ptr = static_cast<int32_t*>(ranks->data);

  bool is_lora_b = n > k;

  if ((!is_lora_b && num_lora_tokens <= SMALL_BATCH_LORA_A_THRESHOLD) ||
      (is_lora_b && num_lora_tokens <= SMALL_BATCH_LORA_B_THRESHOLD)) {
    bool inplace = beta > 0;
    tensorrt_llm::grouped_gemv_lora(static_cast<half*>(x->data),
				    static_cast<half*>(weight->data),
				    static_cast<half*>(out->data),
				    indices_counts_ex_scan_ptr,
				    ranks_ptr,
				    active_slots_ptr,
				    num_lora_tokens, n, k, num_groups, inplace, stream);
    return;
  }

  if (n > k) {
    // LoRA B
    auto func = cutlass_group_gemm_lora_sm80<cutlass::gemm::GemmShape<32, 128, 16>,
                                             cutlass::gemm::GemmShape<32, 64, 16>,
                                             cutlass::gemm::GemmShape<16, 8, 8>>;
    func(static_cast<cutlass::half_t*>(x->data), static_cast<cutlass::half_t*>(weight->data),
         indices_counts_ex_scan_ptr, ranks_ptr, active_slots_ptr,
	 static_cast<uint8_t*>(workspace->data), n, k, num_groups, alpha, beta,
         static_cast<cutlass::half_t*>(out->data), stream);
  } else {
    // LoRA A
    auto func = cutlass_group_gemm_lora_sm80<cutlass::gemm::GemmShape<16, 64, 64>,
                                             cutlass::gemm::GemmShape<16, 16, 64>,
                                             cutlass::gemm::GemmShape<16, 8, 16>,
                                             4>;
    func(static_cast<cutlass::half_t*>(x->data), static_cast<cutlass::half_t*>(weight->data),
         indices_counts_ex_scan_ptr, ranks_ptr, active_slots_ptr,
	 static_cast<uint8_t*>(workspace->data), n, k, num_groups, alpha, beta,
         static_cast<cutlass::half_t*>(out->data), stream);
  }

}

void tvm_cutlass_group_gemm_lora_sm80(NDArray x, NDArray weight, NDArray indices_counts_ex_scan,
				      NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
				      NDArray workspace, NDArray out) {
  return tvm_cutlass_group_gemm_lora_sm80_with_beta(x, weight, indices_counts_ex_scan,
						    ranks, active_slots, num_lora_tokens, 0.0f,
						    workspace, out);
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_lora_fp16_sm80")
    .set_body_typed(tvm_cutlass_group_gemm_lora_sm80);

__global__ void prepare_nested_group_gemm_lora_B_arguments(GroupedGemmArguments arg,
    cutlass::half_t* lora_A_out, cutlass::half_t* lora_B1, cutlass::half_t* lora_B2,
    cutlass::half_t* out, int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots,
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

  arg.stride_A[group_id] = padded_rank * 2;
  arg.stride_B[group_id] = padded_rank;
  arg.stride_D[group_id] = n1 + n2;
}

__global__ void prepare_nested_group_gemm_lora_B_arguments(GroupedGemmArguments arg,
    cutlass::half_t* lora_A_out, cutlass::half_t* lora_B1, cutlass::half_t* lora_B2, cutlass::half_t* lora_B3,
    cutlass::half_t* out, int64_t* indices_counts_ex_scan, int32_t* ranks, int32_t* active_slots,
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

  arg.stride_A[group_id] = padded_rank * 3;
  arg.stride_B[group_id] = padded_rank;
  arg.stride_D[group_id] = n1 + n2 + n3;
}

void nested_group_gemm_lora_B_fp16_sm80(NDArray lora_A_out, Array<NDArray> lora_B_weights, NDArray indices_counts_ex_scan,
                                        NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
                                        double beta, NDArray workspace, NDArray out) {
  auto num_combined = lora_B_weights.size();
  if (num_combined == 1) {
    tvm_cutlass_group_gemm_lora_sm80_with_beta(lora_A_out, lora_B_weights[0], indices_counts_ex_scan,
					       ranks, active_slots, num_lora_tokens, beta, workspace, out);
    return;
  }
  auto padded_rank = lora_B_weights[0]->shape[2];
  int out_size = 0;
  for (auto arr: lora_B_weights) {
    out_size += arr->shape[1];
  }
  auto num_loras = ranks->shape[0];

  CHECK_EQ(lora_A_out->shape[1], padded_rank * num_combined);
  CHECK_EQ(out->shape[1], out_size);
  CHECK_EQ(indices_counts_ex_scan->shape[0], num_loras + 1);

  auto num_groups = num_combined * num_loras;

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  auto workspace_ptr = static_cast<uint8_t*>(workspace->data);
  GroupedGemmArguments arg(workspace_ptr, num_groups);

  auto lora_A_out_ptr = static_cast<cutlass::half_t*>(lora_A_out->data);
  auto out_ptr = static_cast<cutlass::half_t*>(out->data);
  auto indices_counts_ex_scan_ptr = static_cast<int64_t*>(indices_counts_ex_scan->data);
  int32_t* active_slots_ptr = static_cast<int32_t*>(active_slots->data);
  int32_t* ranks_ptr = static_cast<int32_t*>(ranks->data);

  // This does not seem to help
  // if (num_lora_tokens <= SMALL_BATCH_LORA_B_THRESHOLD) {
  //   bool inplace = beta > 0;
  //   std::vector<const half*> lora_B_weights_ptrs;
  //   std::vector<int64_t> out_feature_sizes;
  //   for (auto lora_B: lora_B_weights) {
  //     lora_B_weights_ptrs.push_back(static_cast<const half*>(lora_B->data));
  //     out_feature_sizes.push_back(lora_B->shape[1]);
  //   }

  //   tensorrt_llm::nested_grouped_gemv_lora_B(static_cast<half*>(lora_A_out->data),
  // 					     lora_B_weights_ptrs,
  // 					     static_cast<half*>(out->data),
  // 					     indices_counts_ex_scan_ptr,
  // 					     ranks_ptr, active_slots_ptr,
  // 					     num_lora_tokens, out_feature_sizes,
  // 					     padded_rank, num_loras, inplace, stream);
  //   return;
  // }

  dim3 block(num_combined, num_loras);

  if (num_combined == 2) {
    auto n1 = lora_B_weights[0]->shape[1];
    auto n2 = lora_B_weights[1]->shape[1];
    auto lora_B1_ptr = static_cast<cutlass::half_t*>(lora_B_weights[0]->data);
    auto lora_B2_ptr = static_cast<cutlass::half_t*>(lora_B_weights[1]->data);
    prepare_nested_group_gemm_lora_B_arguments<<<1, block, 0, stream>>>(arg, lora_A_out_ptr, lora_B1_ptr, lora_B2_ptr, out_ptr,
									indices_counts_ex_scan_ptr, ranks_ptr, active_slots_ptr,
									n1, n2, padded_rank);

  } else if (num_combined == 3) {
    auto n1 = lora_B_weights[0]->shape[1];
    auto n2 = lora_B_weights[1]->shape[1];
    auto n3 = lora_B_weights[2]->shape[1];
    auto lora_B1_ptr = static_cast<cutlass::half_t*>(lora_B_weights[0]->data);
    auto lora_B2_ptr = static_cast<cutlass::half_t*>(lora_B_weights[1]->data);
    auto lora_B3_ptr = static_cast<cutlass::half_t*>(lora_B_weights[2]->data);
    prepare_nested_group_gemm_lora_B_arguments<<<1, block, 0, stream>>>(arg, lora_A_out_ptr,
									lora_B1_ptr, lora_B2_ptr, lora_B3_ptr,
									out_ptr, indices_counts_ex_scan_ptr,
									ranks_ptr, active_slots_ptr,
									n1, n2, n3, padded_rank);
  } else {
    LOG(FATAL) << "Unsupported number of outer grouped gemm size: " << num_combined;
  }

  using TileShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstShape = cutlass::gemm::GemmShape<16, 8, 8>;
  run_group_gemm_sm80<TileShape, WarpShape, InstShape>(arg, num_groups, 1.0, beta, stream);
}

// Must be called by call_inplace_packed
NDArray nested_group_gemm_lora_B_inplace_fp16_sm80(NDArray lora_A_out, Array<NDArray> lora_B_weights, NDArray indices_counts_ex_scan,
                                                   NDArray ranks, NDArray active_slots, int64_t num_lora_tokens,
                                                   NDArray workspace, NDArray base_out) {
  nested_group_gemm_lora_B_fp16_sm80(lora_A_out, lora_B_weights, indices_counts_ex_scan, ranks, active_slots,
				     num_lora_tokens, 1.0f, workspace, base_out);
  return base_out;
}

TVM_REGISTER_GLOBAL("cutlass.nested_group_gemm_lora_B_fp16_sm80")
    .set_body_typed(nested_group_gemm_lora_B_fp16_sm80);

TVM_REGISTER_GLOBAL("cutlass.nested_group_gemm_lora_B_inplace_fp16_sm80")
    .set_body_typed(nested_group_gemm_lora_B_inplace_fp16_sm80);
}  // namespace grouped_gemm_sm80

}  // namespace runtime
}  // namespace tvm
