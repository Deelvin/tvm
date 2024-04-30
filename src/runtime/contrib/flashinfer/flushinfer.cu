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

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/decode_attention_decl.cuh>
#include <flashinfer/prefill_attention_decl.cuh>
#include <optional>

using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;
using namespace flashinfer;


#define DISPATCH_TVM_CUDA_DTYPE(dl_dtype, cuda_dtype, ...)   \
  if (dl_dtype.code == kDLFloat && dl_dtype.bits == 16) {    \
    using cuda_dtype = half;                                 \
    __VA_ARGS__                                              \
  } else {                                                   \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code; \
  }

#define DISPATCH_TVM_CUDA_IDTYPE(dl_dtype, cuda_dtype, ...)  \
  if (dl_dtype.code == kDLInt && dl_dtype.bits == 32) {      \
    using cuda_dtype = int32_t;                              \
    __VA_ARGS__                                              \
  } else {                                                   \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code; \
  }


struct FlashInferThreadEntry {
  FlashInferThreadEntry() = default;
  ~FlashInferThreadEntry() {
    release();
  }
  
  enum Mode {
    NoEvaluate = 0,
    Prefill = 1,
    Decode = 2,
    RaggedPrefill = 3,
  };

  void setup(NDArray workspace, NDArray qo_indptr, NDArray kv_indptr, 
             NDArray kv_values, NDArray kv_last_page_len, int64_t num_qo_heads, 
             int64_t num_kv_heads, int64_t head_dim, int64_t page_size, int64_t max_num_pages, 
             Mode mode) {
    _workspace = workspace;
    _qo_indptr = qo_indptr;
    _kv_indptr = kv_indptr;
    _kv_values = kv_values;
    _kv_last_page_len = kv_last_page_len;
    _num_qo_heads = num_qo_heads;
    _num_kv_heads = num_kv_heads;
    _head_dim = head_dim;
    _page_size = page_size;
    _max_num_pages = max_num_pages;

    _mode = mode;
    _batch_size = kv_last_page_len->shape[0];
    // TODO: Add shape verification

    size_t workspace_size_in_bytes = _workspace->shape[0] * _workspace->dtype.bits / 8;
    if (_mode | Prefill) {
      DISPATCH_TVM_CUDA_IDTYPE(_kv_indptr->dtype, dtype_idx, {
        cudaError_t status = _prefill_hdl.BeginForward(
            static_cast<void*>(_workspace->data), workspace_size_in_bytes,
            static_cast<dtype_idx*>(_qo_indptr->data) + _qo_indptr->byte_offset / sizeof(dtype_idx),
            _batch_size, _num_qo_heads, _num_kv_heads, _head_dim);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer prefill BeginForward error " << cudaGetErrorString(status);
        }
      });
    }
    if (_mode | Decode) {
      constexpr PageStorage page_storage = PageStorage::kIndices;
      constexpr QKVLayout kv_layout = QKVLayout::kHND;
      using dtype_in = half;

      DISPATCH_TVM_CUDA_IDTYPE(_kv_indptr->dtype, dtype_idx, {
        cudaError_t status = _decode_hdl.BeginForward<page_storage, kv_layout, dtype_in, dtype_in, dtype_idx>(
                    static_cast<void*>(_workspace->data), workspace_size_in_bytes,
                    static_cast<dtype_idx*>(_kv_indptr->data) +
                        _kv_indptr->byte_offset / sizeof(dtype_idx),
                    static_cast<dtype_idx*>(_kv_last_page_len->data) +
                        _kv_last_page_len->byte_offset / sizeof(dtype_idx),
                    _batch_size, _num_qo_heads, _num_kv_heads, _head_dim, _page_size,
                    PosEncodingMode::kNone);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer decode BeginForward error " << cudaGetErrorString(status);
        }
      });
    }
  }

  void release() {
    _workspace = {};
    _qo_indptr = {};
    _kv_indptr = {};
    _kv_values = {};
    _kv_last_page_len = {};
    
    if (_mode | Prefill) {
      _prefill_hdl.EndForward();
    }
    if (_mode | Decode) {
      _decode_hdl.EndForward();
    }
  }

  static FlashInferThreadEntry* ThreadLocal();

  NDArray _workspace;
  NDArray _qo_indptr;
  NDArray _kv_indptr;
  NDArray _kv_values;
  NDArray _kv_last_page_len;
  int64_t _num_qo_heads;
  int64_t _num_kv_heads;
  int64_t _head_dim;
  int64_t _page_size;
  int64_t _max_num_pages;
  int64_t _batch_size;

  bool _causal = true;  // false - 'TopLeft', true - 'BottomRight'
  PosEncodingMode _pos_mode = PosEncodingMode::kNone;

  Mode _mode;
  BatchPrefillHandler _prefill_hdl;
  BatchDecodeHandler _decode_hdl;
};  // FlashInferThreadEntry

typedef dmlc::ThreadLocalStore<FlashInferThreadEntry> FlashInferThreadStore;

FlashInferThreadEntry* FlashInferThreadEntry::ThreadLocal() { return FlashInferThreadStore::Get(); }


void FlashInferBeforePrefillForward(NDArray workspace_buffer, 
                                       NDArray qo_indptr, 
                                       NDArray kv_indptr, 
                                       NDArray kv_values, 
                                       NDArray kv_last_page_len, 
                                       int64_t num_qo_heads, 
                                       int64_t num_kv_heads, 
                                       int64_t head_dim, 
                                       int64_t page_size, 
                                       int64_t max_num_pages) {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();

  attn_ctx->setup(workspace_buffer, 
                  qo_indptr, 
                  kv_indptr, 
                  kv_values, 
                  kv_last_page_len, 
                  num_qo_heads, 
                  num_kv_heads, 
                  head_dim, 
                  page_size, 
                  max_num_pages, 
                  FlashInferThreadEntry::Mode::Prefill);
}


void FlashInferBeforeDecodeForward(NDArray workspace_buffer, 
                                      NDArray qo_indptr, 
                                      NDArray kv_indptr, 
                                      NDArray kv_values, 
                                      NDArray kv_last_page_len, 
                                      int64_t num_qo_heads, 
                                      int64_t num_kv_heads, 
                                      int64_t head_dim, 
                                      int64_t page_size, 
                                      int64_t max_num_pages) {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();

  attn_ctx->setup(workspace_buffer, 
                  qo_indptr, 
                  kv_indptr, 
                  kv_values, 
                  kv_last_page_len, 
                  num_qo_heads, 
                  num_kv_heads, 
                  head_dim, 
                  page_size, 
                  max_num_pages, 
                  FlashInferThreadEntry::Mode::Decode);
}


void FlashInferBeforeAppend(NDArray append_indptr, 
                            NDArray kv_indptr, 
                            NDArray kv_values, 
                            NDArray kv_last_page_len, 
                            int64_t num_qo_heads, 
                            int64_t num_kv_heads, 
                            int64_t head_dim, 
                            int64_t page_size, 
                            int64_t max_num_pages) {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();

  attn_ctx->setup({}, /*qo_indptr*/
                  append_indptr, 
                  kv_indptr, 
                  kv_values, 
                  kv_last_page_len, 
                  num_qo_heads, 
                  num_kv_heads, 
                  head_dim, 
                  page_size, 
                  max_num_pages, 
                  FlashInferThreadEntry::Mode::NoEvaluate);
}


void FlashInferEndForward() {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();
  attn_ctx->release();
}


void FlashInferForward(NDArray q_data, NDArray kv_cache, NDArray output) {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();
  auto& page_table_indptr = attn_ctx->_kv_indptr;
  auto& page_table_values = attn_ctx->_kv_values;
  auto& last_page_len = attn_ctx->_kv_last_page_len;
  auto& qo_indptr = attn_ctx->_qo_indptr;
  
  double attn_score_scaling_factor = 1.0;
  CHECK_EQ(q_data->device.device_type, kDLCUDA) << "The device of q_data must be CUDA.";
  CHECK_EQ(kv_cache->device.device_type, kDLCUDA) << "The device of kv pages must be CUDA.";
  
  CHECK_EQ(page_table_indptr->device.device_type, kDLCUDA)
      << "The device of page_table_indptr matrix must be CUDA.";
  CHECK_EQ(page_table_values->device.device_type, kDLCUDA)
      << "The device of page_table_values matrix must be CUDA.";
  CHECK_EQ(last_page_len->device.device_type, kDLCUDA)
      << "The device of last_page_len matrix must be CUDA.";
  CHECK_EQ(qo_indptr->device.device_type, kDLCUDA)
      << "The device of qo_indptr matrix must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA) << "The device of output must be CUDA.";
  
  int32_t dev_id = q_data->device.device_id;
  CHECK_EQ(kv_cache->device.device_id, dev_id);
  CHECK_EQ(page_table_indptr->device.device_id, dev_id);
  CHECK_EQ(page_table_values->device.device_id, dev_id);
  CHECK_EQ(last_page_len->device.device_id, dev_id);
  CHECK_EQ(qo_indptr->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && kv_cache->dtype.lanes == 1 && output->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == kv_cache->dtype.bits && q_data->dtype.code == kv_cache->dtype.code);
  CHECK(page_table_indptr->dtype.lanes == 1 && page_table_values->dtype.lanes == 1 &&
        last_page_len->dtype.lanes == 1 && qo_indptr->dtype.lanes == 1);
  CHECK(page_table_indptr->dtype.bits == page_table_values->dtype.bits &&
        page_table_indptr->dtype.bits == last_page_len->dtype.bits &&
        page_table_indptr->dtype.bits == qo_indptr->dtype.bits &&
        page_table_indptr->dtype.code == page_table_values->dtype.code &&
        page_table_indptr->dtype.code == last_page_len->dtype.code &&
        page_table_indptr->dtype.code == qo_indptr->dtype.code);

  CHECK_EQ(kv_cache->ndim, 5);
  CHECK_EQ(kv_cache->shape[1], 2);
  CHECK_EQ(kv_cache->shape[2], attn_ctx->_num_kv_heads);
  CHECK_EQ(q_data->shape[1], attn_ctx->_num_qo_heads);
  CHECK_EQ(kv_cache->shape[4], attn_ctx->_head_dim);
  CHECK_EQ(kv_cache->shape[3], attn_ctx->_page_size);

  CHECK_EQ(last_page_len->ndim, 1);
  int64_t num_total_seqs = last_page_len->shape[0];

  CHECK_EQ(qo_indptr->ndim, 1);
  CHECK_EQ(qo_indptr->shape[0], num_total_seqs + 1);

  CHECK_EQ(page_table_indptr->ndim, 1);
  CHECK_EQ(page_table_indptr->shape[0], num_total_seqs + 1);
  CHECK_EQ(page_table_values->ndim, 1);

  CHECK_EQ(q_data->ndim, 3);
  CHECK_EQ(output->ndim, 3);
  CHECK_EQ(q_data->shape[2], attn_ctx->_head_dim);
  CHECK_EQ(output->shape[1], attn_ctx->_num_qo_heads);
  CHECK_EQ(output->shape[2], attn_ctx->_head_dim);
  
  int64_t nhead_kv = attn_ctx->_num_kv_heads;
  int64_t nhead_qo = attn_ctx->_num_qo_heads;
  int64_t nfeat = attn_ctx->_head_dim;
  int64_t page_size = attn_ctx->_page_size;

  constexpr PageStorage page_storage = PageStorage::kIndices;
  constexpr QKVLayout kv_layout = QKVLayout::kHND;
  const float sm_scale = attn_score_scaling_factor / std::sqrt(static_cast<float>(nfeat));

  DISPATCH_TVM_CUDA_DTYPE(kv_cache->dtype, dtype_in, {
    DISPATCH_TVM_CUDA_DTYPE(output->dtype, dtype_out, {
      DISPATCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
        paged_kv_t<page_storage, kv_layout, dtype_in, dtype_idx> cache(
            nhead_kv, page_size, nfeat, num_total_seqs, static_cast<dtype_in*>(kv_cache->data),
            static_cast<dtype_idx*>(page_table_values->data) +
                page_table_values->byte_offset / sizeof(dtype_idx),
            static_cast<dtype_idx*>(page_table_indptr->data) +
                page_table_indptr->byte_offset / sizeof(dtype_idx),
            static_cast<dtype_idx*>(last_page_len->data) +
                last_page_len->byte_offset / sizeof(dtype_idx),
            nullptr);
        
        if (attn_ctx->_mode == FlashInferThreadEntry::Mode::Prefill) {
          cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<
              page_storage, kv_layout, dtype_in, dtype_out, dtype_idx>(
              &attn_ctx->_prefill_hdl, 
              static_cast<dtype_in*>(q_data->data),
              static_cast<dtype_idx*>(qo_indptr->data) +
                  qo_indptr->byte_offset / sizeof(dtype_idx),
              nullptr,
              cache, 
              static_cast<dtype_out*>(output->data),
              /*lse=*/nullptr, 
              nhead_qo,
              attn_ctx->_causal, 
              attn_ctx->_pos_mode,
              /*allow_fp16_qk_reduction=*/false, 
              sm_scale);
          if (status != cudaSuccess) {
            LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
          }
        } else if (attn_ctx->_mode == FlashInferThreadEntry::Mode::Decode) {
          cudaError_t status = BatchDecodeWithPagedKVCacheWrapper<page_storage, kv_layout,
              dtype_in, dtype_out, dtype_idx>(
            &attn_ctx->_decode_hdl, 
            static_cast<dtype_in*>(q_data->data),
            nullptr,
            cache, 
            static_cast<dtype_out*>(output->data),
            /*lse=*/nullptr, 
            nhead_qo,
            attn_ctx->_pos_mode, 
            sm_scale);
          if (status != cudaSuccess) {
            LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
          }
        }
      })
    })
  });
}


NDArray FlashInferAppendKVCache(NDArray append_k, NDArray append_v, NDArray kv_data) {
  auto attn_ctx = FlashInferThreadEntry::ThreadLocal();
  NDArray append_indptr = attn_ctx->_qo_indptr;
  NDArray kv_indptr  = attn_ctx->_kv_indptr;
  NDArray kv_indices = attn_ctx->_kv_values;
  NDArray kv_last_page_lens = attn_ctx->_kv_last_page_len;

  CHECK_EQ(append_k->device.device_type, kDLCUDA) << "The device of append_k must be CUDA.";
  CHECK_EQ(append_v->device.device_type, kDLCUDA) << "The device of append_v must be CUDA.";
  CHECK_EQ(append_indptr->device.device_type, kDLCUDA) << "The device of append_indptr must be CUDA.";
  CHECK_EQ(kv_data->device.device_type, kDLCUDA) << "The device of kv_data must be CUDA.";
  CHECK_EQ(kv_indptr->device.device_type, kDLCUDA) << "The device of kv_indptr must be CUDA.";
  CHECK_EQ(kv_indices->device.device_type, kDLCUDA) << "The device of kv_indices must be CUDA.";
  CHECK_EQ(kv_last_page_lens->device.device_type, kDLCUDA) << "The device of kv_last_page_lens must be CUDA.";
  
  int32_t dev_id = kv_data->device.device_id;
  CHECK_EQ(append_k->device.device_id, dev_id);
  CHECK_EQ(append_v->device.device_id, dev_id);
  CHECK_EQ(append_indptr->device.device_id, dev_id);
  CHECK_EQ(kv_indptr->device.device_id, dev_id);
  CHECK_EQ(kv_indices->device.device_id, dev_id);
  CHECK_EQ(kv_last_page_lens->device.device_id, dev_id);

  CHECK_EQ(append_k->ndim, 3);
  CHECK_EQ(append_v->ndim, 3);
  CHECK_EQ(append_indptr->ndim, 1);
  CHECK_EQ(kv_data->ndim, 5);
  CHECK_EQ(kv_indptr->ndim, 1);
  CHECK_EQ(kv_indices->ndim, 1);
  CHECK_EQ(kv_last_page_lens->ndim, 1);


  int64_t num_heads = kv_data->shape[2];
  int64_t page_size = kv_data->shape[3];
  int64_t head_dim = kv_data->shape[4];
  
  // CHECK_EQ(s->shape[0], batch_size);
  // CHECK_EQ(s->shape[1], num_heads);
  // CHECK_EQ(v_other->shape[0], batch_size);
  // CHECK_EQ(v_other->shape[1], num_heads);
  // CHECK_EQ(v_other->shape[2], head_dim);
  // CHECK_EQ(s_other->shape[0], batch_size);
  // CHECK_EQ(s_other->shape[1], num_heads);

  int64_t batch_size = kv_last_page_lens->shape[0];

  constexpr PageStorage page_storage = PageStorage::kIndices;
  constexpr QKVLayout kv_layout = QKVLayout::kHND;
                                  
  DISPATCH_TVM_CUDA_DTYPE(kv_data->dtype, c_type, {
    DISPATCH_TVM_CUDA_IDTYPE(kv_indptr->dtype, idtype, {
      paged_kv_t<page_storage, kv_layout, c_type, idtype> paged_kv(
        num_heads, page_size, head_dim, batch_size, static_cast<c_type*>(kv_data->data),
        static_cast<idtype*>(kv_indices->data), static_cast<idtype*>(kv_indptr->data),
        static_cast<idtype*>(kv_last_page_lens->data));

      cudaError_t status = AppendPagedKVCache(
        paged_kv, static_cast<c_type*>(append_k->data), static_cast<c_type*>(append_v->data),
        static_cast<idtype*>(append_indptr->data), /*stream=*/0);
      if (status != cudaSuccess) {
        LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
      }
    })
  });

  return kv_data;
}


Array<NDArray> FlashInferAllocateKVCache(int head_size, int num_layers, int num_heads, 
                                         int block_size, int num_blocks) {
  Array<NDArray> cache;
  int device_id;
  cudaGetDevice(&device_id);

  DLDevice dev{DLDeviceType::kDLCUDA, device_id};

  for (int i = 0; i < num_layers; ++i) {
    NDArray kv_blocks = NDArray::Empty({num_blocks, 2, num_heads, block_size, head_size},
                                       tvm::runtime::DataType::Float(16), dev);
    cache.push_back(kv_blocks);
  }

  return cache;
}

TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.before_prefill_forward").set_body_typed(FlashInferBeforePrefillForward);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.before_decode_forward").set_body_typed(FlashInferBeforeDecodeForward);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.before_append").set_body_typed(FlashInferBeforeAppend);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.forward").set_body_typed(FlashInferForward);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.end_forward").set_body_typed(FlashInferEndForward);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.append_kv_cache").set_body_typed(FlashInferAppendKVCache);
TVM_REGISTER_GLOBAL("tvm.contrib.flashinfer.allocate_kv_cache").set_body_typed(FlashInferAllocateKVCache);
