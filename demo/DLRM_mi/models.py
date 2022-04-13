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

import platform
import subprocess
from collections import namedtuple

from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name


def get_so_ext():
    return "dylib" if platform.system() == "Darwin" else "so"


class CpuInfo:
    def __init__(self) -> None:
        self.name = None
        self.family = None
        self.base_freq = None
        self.num_cores = None
        self.num_sockets = None
        self.num_threads = None
        self.avx2 = None
        self.avx512 = None
        self.hyper_threading = None


def get_cpu_info():
    res = CpuInfo()
    if platform.system() == "Darwin":
        cpu_info = (subprocess.check_output("sysctl -a", shell=True).strip()).decode()
        spl = cpu_info.split('\n')
        for line in spl:
            if line.startswith('machdep.cpu.brand_string'):
                res.name = line.split(":", 1)[1]
            if line.startswith('hw.cpufrequency'):
                res.base_freq = int(line.split(":", 1)[1])
            if line.startswith('hw.physicalcpu'):
                res.num_cores = int(line.split(":", 1)[1])
            if line.startswith('hw.ncpu'):
                res.num_threads = int(line.split(":", 1)[1])
            if line.startswith('hw.packages'):
                res.num_sockets = int(line.split(":", 1)[1])
            if line.startswith('hw.optional.avx2_0'):
                res.avx2 = int(line.split(":", 1)[1])
            if line.startswith('hw.optional.avx512f'):
                res.avx512 = int(line.split(":", 1)[1])

        res.hyper_threading = res.num_cores != res.num_threads

    elif platform.system() == "Linux":
        cpu_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
        spl = cpu_info.split('\n')
        for line in spl:
            if line.startswith('Model name'):
                res.name = line.split(":", 1)[1].lstrip()
            if line.startswith('CPU family'):
                res.family = int(line.split(":", 1)[1].lstrip())
            if line.startswith('CPU MHz'):
                res.base_freq = float(line.split(":", 1)[1].lstrip())
            if line.startswith('CPU(s)'):
                res.num_threads = int(line.split(":", 1)[1].lstrip())
            if line.startswith('Socket(s)'):
                res.num_sockets = int(line.split(":", 1)[1].lstrip())
            if line.startswith('Thread(s) per core'):
                res.hyper_threading = line.split(":", 1)[1].lstrip() == "2"
            if line.startswith('Flags'):
                flags = line.split(":", 1)[1].lstrip()
                flags = flags.split(" ")
                res.avx2 = "avx2" in flags
                res.avx512 = "avx512f" in flags

        res.num_cores = res.num_threads / 2 if res.hyper_threading else res.num_threads

    else:
        res.name = "N/A"
        res.family = 0
        res.base_freq = 1
        res.num_cores = 1
        res.num_threads = 1
        res.num_sockets = 1
        res.avx2 = False
        res.avx512 = False
        res.hyper_threading = False

    return res


def get_host_target():
    info = get_cpu_info()
    cpu_name = info.name
    family = info.family
    isa = get_host_isa()

    if cpu_name.find("AMD") != -1:
        assert isa == "avx2", "Unknown AMD CPU ..."
        # family info from: https://en.wikipedia.org/wiki/List_of_AMD_CPU_microarchitectures
        if family == 25:
            return "llvm -mcpu=znver3"
        elif family in [23, 24]:
            return "llvm -mcpu=znver2"
        else:
            assert False, "Unknown AMD CPU ..."
    elif cpu_name.find("Intel") != -1:
        if isa == "avx2":
            return "llvm -mcpu=core-avx2"
        elif isa == "avx512":
            return "llvm -mcpu=skylake-avx512"
        else:
            assert False, "Unknown Intel CPU ..."

    return "llvm"


def get_host_isa():
    info = get_cpu_info()
    if info.avx512:
        return "avx512"
    if info.avx2:
        return "avx2"
    else:
        return "x86_64"

#other types tbd
scalar_type_to_tvm_type = {
     1 : "float32",         # 1
     2 : "uint8",         # 2 v11?
     7 : "int64",       # 7
}

def get_input(model, batch_size, model_name):
  retval = {}
  shape_dtypes = {}
  for input in model.graph.input:
    nm = input.name
    shape = []
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if tensor_type.HasField("shape"):
      for d in tensor_type.shape.dim:
        if d.HasField("dim_value"): # known dimension
          shape.append(int(d.dim_value))
        elif d.HasField("dim_param"): # unknown dimension with symbolic name
          # workaround for now!
          shape.append(batch_size)
    # print(tensor_type)
    dtype = "float32"
    if tensor_type.HasField('elem_type'):
      # print(sym_help.cast_pytorch_to_onnx.keys())
      if not tensor_type.elem_type in scalar_type_to_tvm_type.keys():
        print("ERROR: unknown onnx dtype : ", tensor_type.elem_type)
        exit(0)
      dtype = scalar_type_to_tvm_type[tensor_type.elem_type]
      # print(tensor_type.elem_type)
    retval[nm] = shape
    shape_dtypes[nm] = dtype
  # workaround because DLRM does not have dynamic batch
  if model_name == "dlrm":
    retval["input.1"][0] = batch_size
    retval["lS_o"][1] = batch_size
    retval["lS_i"][1] = batch_size
  return retval, shape_dtypes


###########################
# DLRM 99 - FP32
###########################

def load_dlrm(model_path, batch_size):
    shape_dict = {
        "input.1": (batch_size, 13),
        "lS_o": (26, batch_size),
        "lS_i": (26, batch_size)
    }

    import onnx
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)

    return mod, params


def inputs_dlrm(batch_size):
    # set random input
    x_in = np.random.randint(0, 100, size=(batch_size, 13)).astype("float32")
    ls_i_in = np.random.randint(0, 100, size=(26, batch_size)).astype("int64")
    ls_o_in = np.array([range(batch_size) for _ in range(26)]).astype("int64")

    return {
        "input.1": x_in,
        "lS_o": ls_i_in,
        "lS_i": ls_o_in,
    }


dyn_batch_config_dlrm = [(0, 0, False), (1, 1, False), (2, 1, False), (0, 0, True)]


###########################
# BERT large - FP32
###########################

def load_bert(model_path, batch_size):
    shape_dict = {
        "input_ids": (batch_size, 384),
        "input_mask": (batch_size, 384),
        "segment_ids": (batch_size, 384),
    }

    import onnx
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)

    return mod, params


def inputs_bert(batch_size):
    ids_in = np.random.randint(0, 100, size=(batch_size, 384)).astype("int64")
    mask_in = np.random.randint(0, 100, size=(batch_size, 384)).astype("int64")
    segment_in = np.random.randint(0, 100, size=(batch_size, 384)).astype("int64")

    return {
        "input_ids": ids_in,
        "input_mask": mask_in,
        "segment_ids": segment_in,
    }


###########################
# BERT large - INT8
###########################

def load_bert_i8(model_path, batch_size):
    # shape_dict = {
    #     "input_ids": (batch_size, 13),
    #     "attention_mask": (26, batch_size),
    #     "token_type_ids": (26, batch_size)
    # }

    # import onnx
    # onnx_model = onnx.load(model_path)
    # mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    # mod = transform.InferType()(mod)
    # mod = transform.DynamicToStatic()(mod)

    # return mod, params
    assert False, "Unimplemented"

def inputs_bert_i8(batch_size):
    assert False, "Unimplemented"


###########################
# ResNet 50 - FP32
###########################

def load_resnet(model_path, batch_size):
    shape_dict = {
        "input_tensor:0": [batch_size, 3, 224, 224],
    }

    import onnx
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)

    return mod, params

def default_load(model_path, batch_size, model_name):

    import onnx
    onnx_model = onnx.load(model_path)
    shape_dict, _ = get_input(onnx_model, batch_size, model_name)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    del onnx_model
    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)

    return mod, params

def inputs_resnet(batch_size):
    x_in = np.random.randint(0, 100, size=[batch_size, 3, 224, 224]).astype("float32")
    return {"input_tensor:0": x_in}


dyn_batch_config_resnet = [(0, 0, False), (0, 0, True), (1, 0, True)]


###########################
# ResNet 50 - INT8
###########################

def load_resnet_i8(model_path, batch_size):
    shape_list = [("X", [batch_size, 3, 224, 224])]

    import torch

    torch_model = torch.jit.load(model_path)
    torch_model.eval()

    mod, params = relay.frontend.from_pytorch(torch_model, shape_list, keep_quantized_weight=True)
    mod["main"] = bind_params_by_name(mod["main"], params)

    return mod, params


def inputs_resnet_i8(batch_size):
    x_in = np.random.randint(0, 100, size=[batch_size, 3, 224, 224]).astype("float32")
    return {"X": x_in}


dyn_batch_config_resnet_i8 = [(0, 0, False), (0, 0, True)]


models = {
    "dlrm":      (load_dlrm,      3, inputs_dlrm,      dyn_batch_config_dlrm),
    "bert":      (load_bert,      4, inputs_bert,      None),
    "bert_i8":   (load_bert_i8,   3, inputs_bert_i8,   None),
    "resnet":    (load_resnet,    2, inputs_resnet,    dyn_batch_config_resnet),  # opt_level=3 lead to change dense->matmul with wrong shapes. Cannot tune
    "resnet_i8": (load_resnet_i8, 3, inputs_resnet_i8, dyn_batch_config_resnet),
}


default_model_path = {
    "resnet": "__models/resnet50_fp32/resnet50_v1.onnx",
    "resnet_i8": "__models/resnet50_int8/resnet50_INT8bit_quantized.pt",
    "bert": "__models/bert_fp32/model.onnx",
    "bert_i8": "__models/bert_int8/bert_large_v1_1_fake_quant.onnx",
    "dlrm": "__models/dlrm_99_fp32/dlrm_s_pytorch_0505.onnx",
}
