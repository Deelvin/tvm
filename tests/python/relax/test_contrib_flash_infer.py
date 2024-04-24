# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import pytest

import tvm.testing
import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


has_flash = tvm.get_global_func("tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache", True)

flash_attn_enabled = pytest.mark.skipif(
    not has_flash,
    reason="Flash attention not enabled.",
)

pytestmark = [flash_attn_enabled] + tvm.testing.requires_cuda.marks()


def process_ref_attention(req_config, num_heads, head_dim, max_num_pages, page_size):
    dev = tvm.cpu()

    # Reference inference via llvm
    num_queries, num_keys = tir.Var("num_queries", "int64"), tir.Var("num_keys", "int64")
    q = relax.Var("q", relax.TensorStructInfo([1, num_queries, num_heads, head_dim], "float16"))
    k = relax.Var("k", relax.TensorStructInfo([1, num_keys, num_heads, head_dim], "float16"))
    v = relax.Var("v", relax.TensorStructInfo([1, num_keys, num_heads, head_dim], "float16"))
    
    bb = relax.BlockBuilder()
    with bb.function("main", [q, k, v]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.attention(q, k, v))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    ex = relax.build(bb.get(), target="llvm") # "cuda"
    vm = relax.VirtualMachine(ex, dev)

    res_data = np.empty(shape=(0, num_heads, head_dim), dtype="float16")
    q_data = np.empty(shape=(0, num_heads, head_dim), dtype="float16")
    kv_data = np.empty(shape=(max_num_pages, 2, page_size, num_heads, head_dim), dtype="float16")

    k_ragged = np.empty(shape=(0, num_heads, head_dim), dtype="float16")
    v_ragged = np.empty(shape=(0, num_heads, head_dim), dtype="float16") 
    kv_ragged_indptr = [0]

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_len = []
    
    cur_page_idx = 0
    cur_qo_indptr = 0
    cur_kv_ragged_indptr = 0

    for num_kv, num_q in req_config:
        num_kv += num_q
        q = np.random.randn(1, num_q, num_heads, head_dim).astype("float16")
        k = np.random.randn(1, num_kv, num_heads, head_dim).astype("float16")
        v = np.random.randn(1, num_kv, num_heads, head_dim).astype("float16")
        
        args = [tvm.nd.array(a, dev) for a in (q, k, v)]
        res = vm["main"](*args).numpy()
        
        # squize nb dimension
        q, k, v, res = (np.squeeze(a, axis=0) for a in (q, k, v, res))

        # store queries and result as ragged tensor 
        cur_qo_indptr += res.shape[0]
        qo_indptr.append(cur_qo_indptr)
        q_data = np.append(q_data, q, axis=0)
        res_data = np.append(res_data, res, axis=0)

        # store VK cache data as ragged tensors  
        k_ragged = np.append(k_ragged, k, axis=0)
        v_ragged = np.append(v_ragged, v, axis=0)
        cur_kv_ragged_indptr += num_kv
        kv_ragged_indptr.append(cur_kv_ragged_indptr)
        
        # store full filed pages 
        while num_kv > page_size:
            kv_data[cur_page_idx, 0] = k[0:page_size]
            kv_data[cur_page_idx, 1] = v[0:page_size]
            kv_indices.append(cur_page_idx)
            k = np.delete(k, np.s_[0:page_size], axis=0)
            v = np.delete(v, np.s_[0:page_size], axis=0)
            cur_page_idx += 1
            num_kv -= page_size

        # store partially filed pages
        kv_data[cur_page_idx, 0, 0:num_kv] = k
        kv_data[cur_page_idx, 1, 0:num_kv] = v
        kv_indices.append(cur_page_idx)
        cur_page_idx += 1
        kv_indptr.append(cur_page_idx)
        kv_last_page_len.append(num_kv)
    
    # FlashInfer uses HND layout: [max_num_pages, 2, num_heads, page_size, head_dim]
    kv_data = np.transpose(kv_data, axes=[0, 1, 3, 2, 4])

    (
        qo_indptr, 
        kv_indptr, 
        kv_indices, 
        kv_last_page_len,
        kv_ragged_indptr
    ) = (np.array(a, dtype="int32") for a in (
        qo_indptr, 
        kv_indptr, 
        kv_indices, 
        kv_last_page_len,
        kv_ragged_indptr
        ))

    return (
        (q_data, qo_indptr, kv_data, kv_indptr, kv_indices, kv_last_page_len), 
        res_data,
        (k_ragged, v_ragged, kv_ragged_indptr)
    )


@I.ir_module
class InferModule:
    @R.function(pure=False)
    def main(
        query: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        qo_indptr: R.Tensor(("num_seqs_plus1",), dtype="int32"),
        kv_cache: R.Tensor(("max_num_pages", 2, "num_heads", "page_size", "head_dim"), dtype="float16"),
        kv_indptr: R.Tensor(("num_seqs_plus1",), dtype="int32"),
        kv_indices: R.Tensor(("kv_num_pages",), dtype="int32"),
        kv_last_page_len: R.Tensor(("num_seqs",), dtype="int32"),
    ) -> R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"):
        with R.dataflow():
            num_seqs = T.int64()
            num_tokens = T.int64()
            head_dim = T.int64()
            num_heads = T.int64()

            workspace_size = T.int64(8 * 1024 * 1024)  # 8 MB 
            
            hdl = T.int64(0)  # handle is in rage [0-7] 
            workspace = R.zeros((workspace_size,), "uint8")
            output = R.zeros((num_tokens, 40, 128), "float16")  # Assume out of attention is the same  

            # TODO: nullptr means dafault stream. Will use dafault stream meanwhile.
            copy_stream = R.call_pure_packed("vm.builtin.nullptr", sinfo_args=[relax.PrimStructInfo("handle")])
            
            # tensor with "dlt->data == mullptr" means do not use this feature 
            lse = R.call_pure_packed("vm.builtin.nullptr_tensor", R.shape([num_tokens]), query, sinfo_args=[R.Tensor()])
            q_offset = R.call_pure_packed("vm.builtin.nullptr_tensor", R.shape([num_tokens]), qo_indptr, sinfo_args=[R.Tensor()])
            k_rope_pos_offset = R.call_pure_packed("vm.builtin.nullptr_tensor", R.shape([num_seqs]), qo_indptr, sinfo_args=[R.Tensor()])
            
            # NOTE: flashinfer.attention_*** functions are not pure, but we can treat it like pure with some allowance:
            #       * To prevent shuffling of order we mark hdl as mutable object.
            #       * Workspace should be hold by _begin_forward and released by _end_forward
            _ = R.call_inplace_packed(
                "flashinfer.attention_kernel_prefill_with_paged_kv_cache_begin_forward",
                hdl, # int64_t handler_idx, 
                workspace, # DLTensor* workspace_buffer, 
                qo_indptr, # DLTensor* qo_indptr, 
                num_seqs, # int64_t batch_size,
                num_heads, # int64_t num_qo_heads, 
                num_heads, # int64_t num_kv_heads, 
                head_dim, # int64_t head_dim, 
                copy_stream, # TVMStreamHandle copy_stream)
                sinfo_args=[],
                inplace_indices=[0]
            )

            _ = R.call_inplace_packed(
                "flashinfer.attention_kernel_prefill_with_paged_kv_cache",                    
                hdl,
                query,
                qo_indptr,
                kv_cache,
                kv_indptr,
                kv_indices,
                kv_last_page_len, 
                k_rope_pos_offset, # DLTensor* k_rope_pos_offset       Can be nullptr
                q_offset, # DLTensor* q_offset                         Can be nullptr
                output, # DLTensor* output,            
                lse, # DLTensor* lse,                                  Can be nullptr
                T.int64(0), # int64_t causal,                          Bool 0=False, 1=True
                T.int64(0), # int64_t pos_encoding_mode,               0=NONE / 1=ROPE_LLAMA / 2=ALIBI
                T.float64(1.0), # double rope_scale = 1.0,             Is not used in case of pos_encoding_mode != ROPE_LLAMA       
                T.float64(1e4), # double rope_theta = 1e4,             Is not used in case of pos_encoding_mode != ROPE_LLAMA       
                T.float64(1.0), # double attn_score_scaling_factor = 1.0f
                sinfo_args=(relax.PrimStructInfo("int64"), R.Tensor(), R.Tensor()),
                inplace_indices=[0, 9, 10],  # hdl, output, lse
            )

            _ = R.call_inplace_packed(
                "flashinfer.attention_kernel_prefill_with_paged_kv_cache_end_forward", 
                hdl,
                sinfo_args=[],
                inplace_indices=[0],
            )
            R.output(output)

        return output


@I.ir_module
class AppendToChacheModule:
    @R.function(pure=False)
    def main(
        append_k: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        append_v: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        append_indptr: R.Tensor(("num_seqs_plus1",), dtype="int32"),
        kv_cache: R.Tensor(("max_num_pages", 2, "num_heads", "page_size", "head_dim"), dtype="float16"),
        kv_indptr: R.Tensor(("num_seqs_plus1",), dtype="int32"),
        kv_indices: R.Tensor(("kv_num_pages",), dtype="int32"),
        kv_last_page_len: R.Tensor(("num_seqs",), dtype="int32"),
    ):
        with R.dataflow():            
            _ = R.call_inplace_packed(
                "flashinfer.append_paged_kv_cache",
                append_k,
                append_v,
                append_indptr,
                kv_cache,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                sinfo_args=[],
                inplace_indices=[3]
            )

        return kv_cache


def test_flash_infere_with_paged_kvcache():
    num_heads = 40
    head_dim = 128

    max_num_pages = 100
    page_size = 32
    
    reqs_config = [  
        # (num_kv, num_q)
        (0, 10),  # prefill reques
        (42, 1),  # decode request
        (67, 10),  # speculative decode check request   
        (40, 13), 
    ]

    np.random.seed(0)
    args, ref_res, kv_ragged = process_ref_attention(reqs_config, num_heads, head_dim, max_num_pages, page_size)

    target = "cuda"

    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(InferModule)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)

    inputs = [tvm.nd.array(inp, dev) for inp in args]
    res = vm["main"](*inputs).numpy()
    
    assert np.allclose(res, ref_res, atol=0.01, rtol=0.01)


def test_flash_infere_append_to_paged_kvcache():
    num_heads = 40
    head_dim = 128

    max_num_pages = 100
    page_size = 32
    
    reqs_config = [  
        # (num_kv, num_q)
        (0, 10),  # prefill reques
        (42, 1),  # decode request
        (67, 10),  # speculative decode check request   
        (40, 13), 
    ]

    np.random.seed(0)
    infer_args, ref_res, kv_ragged = process_ref_attention(reqs_config, num_heads, head_dim, max_num_pages, page_size)

    target = "cuda"

    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(AppendToChacheModule)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    
    append_k, append_v, append_indptr = kv_ragged
    _, _, kv_data_ref, kv_indptr, kv_indices, kv_last_page_len = infer_args
    kv_data = np.empty(shape=(max_num_pages, 2, num_heads, page_size, head_dim), dtype="float16")
    
    inputs = (append_k, append_v, append_indptr, kv_data, kv_indptr, kv_indices, kv_last_page_len)
    inputs = [tvm.nd.array(inp, dev) for inp in inputs]

    kv_data = vm["main"](*inputs).numpy()
    
    # should be identical
    assert np.max(np.abs(kv_data_ref - kv_data)) == 0


if __name__ == "__main__":
    tvm.testing.main()
