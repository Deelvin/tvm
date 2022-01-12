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
This  script set-ups data and environment for DLRM model inference.
"""

import os
import psutil
import sys
from subprocess import Popen, PIPE

file_path = os.path.realpath(__file__)
print(file_path)
demo_folder = os.path.dirname(file_path)

if not 'TVM_HOME' in os.environ:
  print("'TVM_HOME' is not set so the script path is used as reference to the TVM project.")
  tvm_path = os.path.join(demo_folder, "..", "..")
  print(tvm_path)
  os.environ['TVM_HOME']=tvm_path
  sys.path.append(os.path.join(tvm_path, 'python'))

import tvm
import onnx
from tvm import relay, auto_scheduler
from tvm.relay import transform

MODEL_SUFF = 'model'
CONV_SUFF ='converted'
BATCH_SIZE = 128

def load_repo(repo):
  p = Popen(['git', 'clone', repo], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load {} repo\nerror information{}\noutput {}.'.format(repo, err, output))
  return p.returncode

def check_dependencies(subfolder):
  old_path = os.getcwd()
  os.chdir(os.path.join(old_path, subfolder))
  p = Popen(['git', 'submodule', 'update', '--init', '--recursive'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load dependencies for {} subproject\nerror information{}\noutput {}.'.format(subfolder, err, output))
  os.chdir(os.path.join(old_path))

old_path = os.getcwd()
os.chdir(demo_folder)
hdd = psutil.disk_usage(demo_folder)
free_space =  (hdd.free / (2**30))
if free_space < 240:
  print('WARNING: the disk free size is {} and it may not be enough to run full DLRM model.'.format(free_space))

# ret = load_repo('https://github.com/facebookresearch/dlrm.git')
# if ret == 0:converted.git')
# if ret == 0:
#   check_dependencies('inference')
DLRM_DIR = os.path.join(demo_folder, 'dlrm')

# temp_dir = os.path.join(demo_folder, 'inference', 'loadgen')
# os.chdir(temp_dir)
# curr_env = os.environ
# curr_env['CFLAGS'] = "-std=c++14"
# p = Popen(['python', 'setup.py', 'develop', '--user'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=curr_env)
# output, err = p.communicate()
# if p.returncode != 0:
#   print('ERROR: load gen compilation issue {}.'.format(err))

os.chdir(demo_folder)
if os.path.isdir(MODEL_SUFF) != True:
  os.mkdir(MODEL_SUFF)

MODEL_DIR = os.path.join(demo_folder, MODEL_SUFF)

# os.chdir(MODEL_DIR)
# os.system('wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.onnx.tar')
# if os.path.isfile(os.path.join(MODEL_SUFF, 'tb00_40M.onnx.tar')) != True:
#   print("ERROR: cannot find onnx model archive.")
# else:
#   os.system('tar -xvf tb00_40M.onnx.tar')

os.chdir(demo_folder)

# extract weights
onnx_file = os.path.join(MODEL_DIR, 'dlrm_s_pytorch_0505.onnx')

onnx_file = '/home/sshtin/dev/models/dlrm_s_pytorch_0505.onnx'
onnx_model = onnx.load(onnx_file)

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

if os.path.isdir(CONV_SUFF) != True:
  os.mkdir(CONV_SUFF)

ctx = tvm.cpu(0)
target = "llvm"
target_host = "llvm"

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
with tvm.transform.PassContext(opt_level=3, config={}):
    json, lib, param = relay.build(mod, target=target, params=params)
    outPth = os.path.join(demo_folder, CONV_SUFF)
    for key, val in  param.items():
        npData = val.asnumpy()
        # npData.tofile(os.path.join(outPth, key))
        print(os.path.join(outPth, key))
os.chdir(old_path)
