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

# table for dense values: d2 , d3, see mac_count.cc, dense
dense_mats =[
  13  * 512,
  512 * 256,
  256 * 128,
  479 * 1024,
  1024* 1024,
  1024* 512,
  512 * 256,
  256 * 1
]

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input timings log")
parser.add_argument("--output", required=True, help="output csv file")

args = parser.parse_args()
dnnl_codegen = tvm.support.libinfo().get("USE_DNNL_CODEGEN", "OFF")

fname = args.input
ofname = args.output
batch_maps = {}
total_maps = []
batch_size = 0
curr_map = []
iters_map = []
with open(fname) as f:
  lines = f.readlines()
  for line in lines:
    spl = line.split()
    if line.find('BATCH_SIZE') == 0:
      batch_size = int(spl[2])
      continue
    if line.find('runs :') == 0:
      iters = int(spl[-2]) + 2
      print("batch_size = ", batch_size, ", iters = ", iters)
      batch_maps[batch_size] = curr_map.copy()
      curr_map = []
      iters_map.append(iters)
      continue
    if line.find('tvmgen_default_') == 0 or line.find('reshape_nop') == 0:
      curr_map.append([spl[0], float(spl[1])])
      continue
    if line.find('total :') == 0:
      total_maps.append(float(spl[-2]))
print(total_maps)
table_1 = {}
others = [0]*len(total_maps)
embedding = [0]*len(total_maps)
dense = [0]*len(total_maps)
concat = [0]*len(total_maps)
macs_vals = [0]*len(total_maps)
macs_counter = {}
pos = 0
dense_sum = ['fused_nn_dense_add_', 'fused_nn_batch_matmul']
dense_name = 'fused_nn_dense_'

# batch_size = 128
denses = []
for i in range(len(dense_mats)):
  denses.append('dense_{}'.format(i))
# theory = 'theory'
# macs_counter[theory] = []
theory_val = 3.1 * 8 * 2 * 32 /100 # frequency(estimation) * simd width * FMA throughput * threads , division by 100 is to get %
if dnnl_codegen == "ON":
  dense_sum = ['fused_nn_batch_matmul', "tvmgen_default_dnnl_main"]
  dense_name = 'tvmgen_default_dnnl_main'
for key, vals in batch_maps.items():
  dense_counter = 0
  # macs_counter[key] = []
  batch_size = int(key)
  # macs_counter[theory].append(3.1 * 8 * 2 * 32)# frequency(estimation) * simd width * FMA throughput * threads 
  for val in vals:
    if val[0] not in table_1.keys():
      table_1[val[0]] = []
      
    table_1[val[0]].append(val[1])
    if val[0].find('fused_sum') != -1 or val[0].find('fused_take_reshape_take') != -1:
      embedding[pos] += val[1]
      continue
    if val[0].find('_concatenate') != -1:
      concat[pos] += val[1]
      others[pos] += val[1]
      continue
    temp = [x for x in dense_sum if val[0].find(x) != -1]
    if len(temp) > 0:
      dense[pos] += val[1]
      print(val[0])
      if val[0].find(dense_name) != -1:
        nm = "dense_{}".format(dense_counter)
        if nm not in macs_counter.keys():
          macs_counter[nm] = []
        macs = batch_size * dense_mats[dense_counter]
        macs_counter[nm].append(macs/val[1]/1000/theory_val)
        dense_counter += 1
    else:
      others[pos] += val[1]
  pos +=1
batches = list(batch_maps.keys())


print(batches)
thrs = make_str('', batches)
durs = []
eff = []
for i in range(min(len(total_maps), len(iters_map))):
  durs.append(total_maps[i] * iters_map[i]/1000000.)
  eff.append(total_maps[i]/batches[i])
print(iters_map)
print(total_maps)
print(durs)
with open(ofname,"w") as f:
  f.write(thrs)
  for key, vals in table_1.items():
    f.write(make_str(key, vals))
  f.write(thrs)
  f.write(make_str("dense", dense))
  f.write(make_str("embeddings", embedding))
  # f.write(make_str("concat", concat))
  f.write(make_str("others", others))
  f.write(thrs)
  f.write(make_str("efficiency", eff))
  f.write(thrs)
  f.write(make_str("approx dur", durs))
  f.write(thrs)
  for k, vals in macs_counter.items():
    f.write(make_str(k, vals))
print(macs_counter)