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


has_flashinfer = tvm.get_global_func("tvm.contrib.flashinfer.forward", True)

flashinfer_enabled = pytest.mark.skipif(
    not has_flashinfer,
    reason="Flashinfer is not enabled.",
)

pytestmark = [flashinfer_enabled] + tvm.testing.requires_cuda.marks()


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
            lv0 = bb.emit(relax.op.nn.attention(q, k, v, causal_mask="BottomRight"))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    ex = relax.build(bb.get(), target="llvm") # "cuda"
    vm = relax.VirtualMachine(ex, dev)

    res_data = np.empty(shape=(0, num_heads, head_dim), dtype="float16")
    q_data = np.empty(shape=(0, num_heads, head_dim), dtype="float16")
    kv_data = np.zeros(shape=(max_num_pages, 2, page_size, num_heads, head_dim), dtype="float16")

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
class InferRXModule:
    @R.function
    def main(
        query: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        kv_cache: R.Tensor(("max_num_pages", 2, "num_heads", "page_size", "head_dim"), dtype="float16"),
    ) -> R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"):
        with R.dataflow():
            num_tokens = T.int64()
            head_dim = T.int64()
            num_heads = T.int64()
            out = R.call_dps_packed(
                "tvm.contrib.flashinfer.forward", 
                (query, kv_cache), 
                out_sinfo=R.Tensor((num_tokens, num_heads, head_dim), dtype="float16")
            )
            R.output(out)

        return out

@I.ir_module
class AppendToChacheRXModule:
    @R.function
    def main(
        append_k: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        append_v: R.Tensor(("num_tokens", "num_heads", "head_dim"), dtype="float16"),
        kv_cache: R.Tensor(("max_num_pages", 2, "num_heads", "page_size", "head_dim"), dtype="float16"),
    ) -> R.Tensor(("max_num_pages", 2, "num_heads", "page_size", "head_dim"), dtype="float16"):
        with R.dataflow():
            _ = R.call_inplace_packed(
                "tvm.contrib.flashinfer.append_kv_cache", 
                append_k, append_v, kv_cache,
                sinfo_args=[],
                inplace_indices=[2]
            )

        return kv_cache


def test_flashinfer_multiquery():
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
    args, ref_res, _ = process_ref_attention(reqs_config, num_heads, head_dim, max_num_pages, page_size)

    target = "cuda"
    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(InferRXModule)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)

    args = [tvm.nd.array(inp, dev) for inp in args]
    q_data, qo_indptr, kv_data, kv_indptr, kv_indices, kv_last_page_len = args
    workspace = tvm.nd.empty([8*1024*1024], dtype="uint8", device=dev)

    before_forward = tvm.get_global_func("tvm.contrib.flashinfer.before_prefill_forward")
    end_forward = tvm.get_global_func("tvm.contrib.flashinfer.end_forward")

    before_forward(workspace, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, num_heads, num_heads, head_dim, page_size, max_num_pages)
    res = vm["main"](q_data, kv_data).numpy()
    end_forward()
    
    assert np.allclose(res, ref_res, atol=0.01, rtol=0.01)


def test_flashinfer_decode():
    num_heads = 40
    head_dim = 128

    max_num_pages = 100
    page_size = 32
    
    reqs_config = [  
        # (num_kv, num_q)
        (0, 1), 
        (42, 1),  # decode request
        (67, 1),  # speculative decode check request   
        (40, 1), 
    ]

    np.random.seed(0)
    args, ref_res, _ = process_ref_attention(reqs_config, num_heads, head_dim, max_num_pages, page_size)

    target = "cuda"
    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(InferRXModule)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)

    args = [tvm.nd.array(inp, dev) for inp in args]
    q_data, qo_indptr, kv_data, kv_indptr, kv_indices, kv_last_page_len = args
    workspace = tvm.nd.empty([8*1024*1024], dtype="uint8", device=dev)

    # Infer
    before_forward = tvm.get_global_func("tvm.contrib.flashinfer.before_decode_forward")
    end_forward = tvm.get_global_func("tvm.contrib.flashinfer.end_forward")

    before_forward(workspace, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, num_heads, num_heads, head_dim, page_size, max_num_pages)
    res = vm["main"](q_data, kv_data).numpy()
    end_forward()
    
    assert np.allclose(res, ref_res, atol=0.01, rtol=0.01)


def test_flashinfer_append():
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
    args, _, kv_cache_ragged = process_ref_attention(reqs_config, num_heads, head_dim, max_num_pages, page_size)

    target = "cuda"
    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(AppendToChacheRXModule)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)

    _, _, ref_res, kv_indptr, kv_indices, kv_last_page_len = args
    append_k, append_v, append_indptr = kv_cache_ragged
    kv_data = np.zeros((max_num_pages, 2, num_heads, page_size, head_dim), dtype="float16")
    workspace = tvm.nd.empty([8*1024*1024], dtype="uint8", device=dev)

    # Infer
    before_forward = tvm.get_global_func("tvm.contrib.flashinfer.before_prefill_forward")
    end_forward = tvm.get_global_func("tvm.contrib.flashinfer.end_forward")
    
    def to_device(*args):
        return (tvm.nd.array(arg, dev) for arg in args)

    before_forward(workspace, *to_device(append_indptr, kv_indptr, kv_indices, kv_last_page_len), num_heads, num_heads, head_dim, page_size, max_num_pages)
    res = vm["main"](*to_device(append_k, append_v, kv_data)).numpy()
    end_forward()

    assert np.allclose(res, ref_res, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    tvm.testing.main()
