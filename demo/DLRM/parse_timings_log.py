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
This  script perfroms parsing of data colleected by timings.sh script.
"""

import os
import sys
import argparse
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

def make_str(key, vals):
  strO = key + ", "
  for j in vals:
    strO += "{}, ".format(j)
  strO += "\n"
  return strO

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input timings log")
parser.add_argument("--output", required=True, help="output csv file")

args = parser.parse_args()
dnnl_codegen = tvm.support.libinfo().get("USE_DNNL_CODEGEN", "OFF")

fname = args.input
ofname = args.output
time_maps = {}
threads = 0
curr_map = []
with open(fname) as f:
  lines = f.readlines()
  for line in lines:
    if line.find('current threads') == 0:
      spl = line.split()
      if len(curr_map) != 0 and threads != 0:
        time_maps[threads] = curr_map.copy()
        curr_map = []
      threads = int(spl[-1])
    if line.find('tvmgen_default_') == 0 or line.find('reshape_nop') == 0:
      xx = line.split()
      curr_map.append([xx[0], float(xx[1])])
time_maps[threads] = curr_map.copy()
table_1 = {}
others = [0]*len(time_maps.keys())
embedding = [0]*len(time_maps.keys())
dense = [0]*len(time_maps.keys())
concat = [0]*len(time_maps.keys())
pos = 0
dense_sum = ['fused_nn_dense_add_', 'fused_nn_batch_matmul']
if dnnl_codegen == "ON":
  dense_sum = ['fused_nn_batch_matmul', "tvmgen_default_dnnl_main"]
for key, vals in time_maps.items():
  for val in vals:
    if val[0] not in table_1.keys():
      table_1[val[0]] = []
    table_1[val[0]].append(val[1])
    if val[0].find('fused_sum') != -1 or val[0].find('fused_take_reshape_take') != -1:
      embedding[pos] += val[1]
    else:
      if val[0].find('_concatenate') != -1:
        concat[pos] += val[1]
      else:
        temp = [x for x in dense_sum if val[0].find(x) != -1]
        if len(temp) > 0:
          dense[pos] += val[1]
        else:
          others[pos] += val[1]
  pos +=1

thrs = make_str('', time_maps.keys())
with open(ofname,"w") as f:
  f.write(thrs)
  for key, vals in table_1.items():
    f.write(make_str(key, vals))
  f.write(thrs)
  f.write(make_str("dense", dense))
  f.write(make_str("embeddings", embedding))
  f.write(make_str("concat", concat))
  f.write(make_str("others", others))
