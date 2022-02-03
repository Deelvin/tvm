# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
import tvm
import onnx
import numpy as np
import collections
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm.relay import transform
from tvm.relay.op.contrib.register import get_pattern_table

sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

class BERT_TVM_SUT():
    def __init__(self, args):
        target = "llvm -mcpu=znver3"
        ctx = tvm.cpu(0)
        self.batch_size = 1 # 
        dnnl_codegen = tvm.support.libinfo().get("USE_DNNL_CODEGEN", "OFF")
        self.quantized = args.quantized
        if self.quantized:
          name = 'build/data/bert_tf_v1_1_large_fp32_384_v2/bert_large_v1_1_fake_quant.onnx'
          if dnnl_codegen == "OFF":
            seq = tvm.transform.Sequential(
              [
                transform.InferType(),
                transform.FoldConstant(),
                transform.SimplifyInference(),
                transform.FoldScaleAxis(),
                transform.DynamicToStatic(),
                transform.AlterOpLayout(),
                transform.FakeQuantizationToInteger(),
                transform.PartitionGraph(),
              ]
            )
          else:
            patternTBL = get_pattern_table("dnnl")
            seq = tvm.transform.Sequential(
              [
                transform.InferType(),
                transform.FoldConstant(),
                transform.SimplifyInference(),
                transform.FoldScaleAxis(),
                transform.DynamicToStatic(),
                transform.AlterOpLayout(),
                transform.FakeQuantizationToInteger(),
                transform.MergeComposite(patternTBL),
                transform.AnnotateTarget("dnnl"),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
              ]
            )
        else:
          name = 'build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx'
          seq = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.FoldConstant(),
                transform.SimplifyInference(),
                transform.FoldScaleAxis(),
                transform.DynamicToStatic(),
                transform.AlterOpLayout(),
                transform.PartitionGraph(),
            ]
          )
        print("Loading onnx model to convert to TVM model...")
        onnx_model = onnx.load(name)
        shape_dict = self.getInput(onnx_model)
        self.max_seq_length = shape_dict['input_ids'][1]
        self.start_logits = np.ones(self.max_seq_length) * -10000.0
        self.end_logits = np.ones(self.max_seq_length) * -10000.0

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        mod = seq(mod)
        print(mod)
        with tvm.transform.PassContext(opt_level=4, config={}):
          json, lib, param = relay.build(mod, target=target, params=params)
        self.model = graph_executor.create(json, lib, ctx)
        self.model.set_input(**param)
        del(onnx_model)
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")
        self.qsl = get_squad_QSL(args.max_examples)
        self.results = []

    def getInput(self, model):
        retval = {}
        for input in model.graph.input:
            nm = input.name
            shape = []
            # # get type of input tensor
            tensor_type = input.type.tensor_type
            # check if it has a shape:
            if (tensor_type.HasField("shape")):
                for d in tensor_type.shape.dim:
                    if (d.HasField("dim_value")): # known dimension
                        shape.append(int(d.dim_value))
                    elif (d.HasField("dim_param")): # unknown dimension with symbolic name
                        # workaround for now!
                        shape.append(self.batch_size)
            retval[nm] = shape
        return retval

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            self.model.set_input("input_ids", np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :])
            if self.quantized:
              self.model.set_input("attention_mask", np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :])
              self.model.set_input("token_type_ids", np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :])
            else:
              self.model.set_input("input_mask", np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :])
              self.model.set_input("segment_ids", np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :])

            self.model.run()

            start_logitsND = self.model.get_output(0)
            end_logitsND = self.model.get_output(1)
            output = np.stack([start_logitsND.asnumpy(), end_logitsND.asnumpy()], axis=-1)[0]
            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_tvm_sut(args):
    return BERT_TVM_SUT(args)
