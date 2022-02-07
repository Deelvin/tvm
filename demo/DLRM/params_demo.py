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

"""
This  script contains demo configuration parameters.
"""

import subprocess
MODEL_SUFF = 'model'
CONV_SUFF = 'converted'
TEST_DATA_SUFF = 'test_data'
ONNX_FILE_NAME = 'dlrm_s_pytorch_0505.onnx'

BATCH_SIZE = 128
ITERATIONS = 3000

shape_dict = {
    "input.1": (BATCH_SIZE, 13),
    "lS_o": (26, BATCH_SIZE),
    "lS_i": (26, BATCH_SIZE)
}
dtype_dict = {
    "input.1": "float32",
    "lS_o": "int64",
    "lS_i": "int64"
}

def getCPUVendor():
    cpu_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    spl = cpu_info.split('\n')
    print(len(spl))
    for i in range(len(spl)):
        if spl[i].find('Model name') != -1:
            print(spl[i])
            if spl[i].find('AMD') != -1:
                target = "llvm -mcpu=znver3"
                target_host = "llvm -mcpu=znver3"
            else:
                target = "llvm -mcpu=cascadelake"
                target_host = "llvm -cascadelake"
            return target, target_host
