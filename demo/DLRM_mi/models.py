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

import onnx
import torch
import numpy as np

import platform
import subprocess
from collections import namedtuple

from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name


def get_so_ext():
    return "dylib" if platform.system() == "Darwin" else "so"


CpuInfo = namedtuple("CpuInfo", ["name", "base_freq", "num_cores", "num_sockets", "num_threads", "avx2", "avx512"])


def get_cpu_info():
    res = CpuInfo
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

        return res
    else:
        cpu_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
        spl = cpu_info.split('\n')
        for line in spl:
            if line.startswith('Model name'):
                res.name = line.split(":", 1)[1]
    return res


def get_host_target():
    info = get_cpu_info()
    cpu_name = info.name
    if cpu_name.find("AMD") != -1:
        return "llvm -mcpu=znver3"
    else:
        return "llvm -mcpu=core-avx2"


###########################
# DLRM 99 - FP32
###########################

def load_dlrm(model_path, batch_size):
    shape_dict = {
        "input.1": (batch_size, 13),
        "lS_o": (26, batch_size),
        "lS_i": (26, batch_size)
    }

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
    assert False, "Unimplemented"


def inputs_bert(batch_size):
    assert False, "Unimplemented"


###########################
# ResNet 50 - FP32
###########################

def load_resnet(model_path, batch_size):
    shape_dict = {
        "input_tensor:0": [batch_size, 3, 224, 224],
    }

    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

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
    "dlrm": (load_dlrm, inputs_dlrm, dyn_batch_config_dlrm),
    "bert": (load_bert, inputs_bert, None),
    "resnet50": (load_resnet, inputs_resnet, dyn_batch_config_resnet),
    "resnet50_i8": (load_resnet_i8, inputs_resnet_i8, dyn_batch_config_resnet),
}
