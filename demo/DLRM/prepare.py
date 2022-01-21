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
from params_demo import *

file_path = os.path.realpath(__file__)
demo_folder = os.path.dirname(file_path)

if not 'TVM_HOME' in os.environ:
  print("'TVM_HOME' is not set so the script path is used as reference to the TVM project.")
  tvm_path = os.path.join(demo_folder, "..", "..")
  os.environ['TVM_HOME']=tvm_path
else:
  tvm_path = os.environ['TVM_HOME']
sys.path.append(os.path.join(tvm_path, 'python'))

import tvm
import onnx
from tvm import relay
from shutil import copy

def load_repo(repo):
  p = Popen(['git', 'clone', repo], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load {} repo\nerror information{}\noutput {}.'.format(repo, err, output))
  return p.returncode

def check_dependencies(subfolder, commit):
  old_path = os.getcwd()
  os.chdir(os.path.join(old_path, subfolder))
  p = Popen(['git', 'submodule', 'update', '--init', '--recursive'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load dependencies for {} subproject\nerror information{}\noutput {}.'.format(subfolder, err, output))
  p = Popen(['git', 'checkout', commit], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load dependencies for {} subproject\nerror information{}\noutput {}.'.format(subfolder, err, output))
  os.chdir(os.path.join(old_path))

def apply_patches_to_inference():
  run_local_patch = \
'''
diff --git a/recommendation/dlrm/pytorch/run_local.sh b/recommendation/dlrm/pytorch/run_local.sh
index 0d054c6..aaac93a 100755
--- a/recommendation/dlrm/pytorch/run_local.sh
+++ b/recommendation/dlrm/pytorch/run_local.sh
@@ -1,5 +1,10 @@
 #!/bin/bash
 
+export TVM_HOME={0}
+export PYTHONPATH=$TVM_HOME/python:${{PYTHONPATH}}
+export DATA_DIR={2}/recommendation/dlrm/pytorch/tools/fake_criteo/
+export MODEL_DIR={1}/model
+export DLRM_DIR={1}
 source ./run_common.sh
 
 common_opt="--mlperf_conf ../../../mlperf.conf"
'''.format(tvm_path, demo_folder, os.path.join(demo_folder, 'inference'))

  copy(os.path.join('patches', 'backend_tvm.py'), os.path.join('inference', 'recommendation', 'dlrm', 'pytorch', 'python'))
  with open(os.path.join('patches', '_run_local_sh.patch'), 'w') as f:
    f.write(run_local_patch)

  patches = ['_main_py.patch', '_run_common_sh.patch', '_run_local_sh.patch']
  for p in patches:
    copy(os.path.join('patches', p), 'inference')
  old_path = os.getcwd()
  os.chdir(os.path.join(old_path, 'inference'))
  for p in patches:
    p = Popen(['git', 'apply', p], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if p.returncode != 0:
      print('ERROR: cannot apply patch {} for MLPerf inference.'.format(p, err, output))
  os.chdir(os.path.join('recommendation', 'dlrm', 'pytorch', 'tools'))
  p = Popen(['./make_fake_criteo.sh', 'terabyte'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print('ERROR: cannot load dependencies for {} subproject\nerror information{}\noutput {}.'.format(subfolder, err, output))

  os.chdir(old_path)

old_path = os.getcwd()
os.chdir(demo_folder)
hdd = psutil.disk_usage(demo_folder)
free_space =  (hdd.free / (2**30))
if free_space < 240: # just estimation
  print('WARNING: the disk free size is {} GB and it may not be enough to run full DLRM model.'.format(free_space))

print("---- Load required modules ----")
if os.path.exists(os.path.join(demo_folder, 'dlrm')) == True:
  print("WARNING: folder 'dlrm' already exist.")
else:
  ret = load_repo('https://github.com/facebookresearch/dlrm.git')
  if ret == 0:
    check_dependencies('dlrm', '9c2fda79afbc09e277c17e420bffe510125b4f70') # the pipeline is tested on this commit
if os.path.exists(os.path.join(demo_folder, 'inference')) == True:
  print("WARNING: folder 'inference' already exist.")
else:
  ret = load_repo('https://github.com/mlcommons/inference.git')
  if ret == 0:
    check_dependencies('inference', '86d0b16180b40dfd80ebef6f23c26a9853d02ef4')
    apply_patches_to_inference()

DLRM_DIR = os.path.join(demo_folder, 'dlrm')
print("---- Compile loadgen ----------")
temp_dir = os.path.join(demo_folder, 'inference', 'loadgen')
os.chdir(temp_dir)
curr_env = os.environ
curr_env['CFLAGS'] = "-std=c++14"
p = Popen(['python', 'setup.py', 'develop', '--user'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=curr_env)
output, err = p.communicate()
if p.returncode != 0:
  print('ERROR: loadgen compilation issue {}.'.format(err))

os.chdir(demo_folder)
if os.path.isdir(MODEL_SUFF) != True:
  os.mkdir(MODEL_SUFF)

MODEL_DIR = os.path.join(demo_folder, MODEL_SUFF)

os.chdir(MODEL_DIR)
print("---- Load DLRM model ----------")
os.system('wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.onnx.tar')
if os.path.isfile(os.path.join(MODEL_DIR, 'tb00_40M.onnx.tar')) != True:
  print("ERROR: cannot find onnx model archive.")
else:
  os.system('tar -xvf tb00_40M.onnx.tar')

os.chdir(demo_folder)

onnx_file = os.path.join(MODEL_DIR, ONNX_FILE_NAME)
onnx_model = onnx.load(onnx_file)

if os.path.isdir(CONV_SUFF) != True:
  os.mkdir(CONV_SUFF)

ctx = tvm.cpu(0)

print("---- Extract weights ----------")
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
with tvm.transform.PassContext(opt_level=3, config={}):
    json, lib, param = relay.build(mod, target=target, params=params)
    outPth = os.path.join(demo_folder, CONV_SUFF)
    for key, val in  param.items():
        npData = val.asnumpy()
        npData.tofile(os.path.join(outPth, key))
os.chdir(old_path)
