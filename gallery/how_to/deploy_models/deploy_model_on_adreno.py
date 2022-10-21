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
.. _tutorial-deploy-model-on-adreno:
Deploy the Pretrained Model on Adreno
=======================================
**Author**: `Daniil Barinov`_
This article is a step-by-step tutorial to deploy pretrained MXNet VGG16 model on Adreno (on different precisions).
"""

######################################################################
# For us to begin with, MXNet package must be installed.
# A quick solution is to install it via pip:
# .. code-block:: bash
#
#   pip install mxnet
#
# Besides that, you should have TVM builded for Android.
# Please, refer to: `Compile and deploy models on Adreno GPU <link>`_ .
# After the build section there should be two files in *build* directory «libtvm_runtime.so» and «tvm_rpc». Let's push them to the device and run TVM RPC Server.
#################################################################
# TVM RPC Server
# -----------------
# To get the hash of the device use:
# .. code-block:: bash
#
#   adb devices
#
# Then to upload these two files to the device you should use:
# .. code-block:: bash
#
#   adb -s <device_hash> push {libtvm_runtime.so,tvm_rpc} /data/local/tmp_
#
# At this moment you will have «libtvm_runtime.so» and «tvm_rpc» on path /data/local/tmp on your device.
# Sometimes cmake can’t find «libc++_shared.so». Use:
# .. code-block:: bash
#
#   find ~/Android/sdk/ndk/ -name libc++_shared.so
#
# (Linux)
# .. code-block:: bash
#
#   find ~/Library/Android/sdk/ndk/ -name libc++_shared.so
#
# (MacOS)
# to find it and also push it with adb on the desired device:
# .. code-block:: bash
#
#   adb -s <device_hash> push libc++_shared.so /data/local/tmp
#
# We are now ready to run the TVM RPC Server.
# Launch rpc_tracker with following line in 1st console:
# .. code-block:: bash
#
#   python3 -m tvm.exec.rpc_tracker --port 9190
#
# Then we need to run tvm_rpc server from under the desired device in 2nd console:
# .. code-block:: bash
#
#   adb -s <device_hash> reverse tcp:9190 tcp:9190
#   adb -s <device_hash> forward tcp:9090 tcp:9090
#   adb -s <device_hash> forward tcp:9091 tcp:9091
#   adb -s <device_hash> forward tcp:9092 tcp:9092
#   adb -s <device_hash> forward tcp:9093 tcp:9093
#   adb -s <device_hash> shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=9090 --tracker=127.0.0.1:9190 --key=android --port-end=9190
#
# Before proceeding to compile and infer model, specify tracker host and tracker port, make sure that rpc.tracker and tvm_rpc server are running and check that your device is available:
# .. code-block:: bash
#
#   export TVM_TRACKER_HOST=0.0.0.0
#   export TVM_TRACKER_PORT=9190
#
# Check that the tracker is running and the device is available
# .. code-block:: bash
#
#     python -m tvm.exec.query_rpc_tracker --port 9190
#
# For example, if we have 1 Android device.
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    android      1      1     0
#    ----------------------------------


######################################################################
# Load pretrained MXNet model
# ---------------------------
# Create a Relay graph from a MXNet VGG16 model
import  os
import  numpy  as  np
import  mxnet.gluon  as  gluon
import  tvm
from  tvm  import  relay, rpc
from  tvm.contrib  import  utils, ndk
from  tvm.contrib  import  graph_executor

name = "vgg16"
model = gluon.model_zoo.vision.get_model(name, pretrained=True)
input_name = "data"
input_shape = (1, 3, 224, 224)
shape_dict = {input_name: input_shape}
mod, params = relay.frontend.from_mxnet(model, shape_dict)

######################################################################
# Precisions
# ----------
# Since TVM support Mixed Precision, we need to register mixed_precision_conversion:
from  tvm.relay.op  import  register_mixed_precision_conversion

conv2d_acc = "float32"

@register_mixed_precision_conversion("nn.conv2d", level=11)
def  conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
	global  conv2d_acc
	return [
		relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
		conv2d_acc,
		mixed_precision_type,
	] 

@register_mixed_precision_conversion("nn.dense", level=11)
def  conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
	global  conv2d_acc
	return [
		relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
		conv2d_acc,
		mixed_precision_type,
	]
 
######################################################################
# and also the conversion function itself
def  convert_to_dtype(mod, dtype):
	# downcast to float16
	if  dtype == "float16"  or  dtype == "float16_acc32":
		global  conv2d_acc
		conv2d_acc = "float16"  if  dtype == "float16"  else  "float32"
		from  tvm.ir  import  IRModule
		mod = IRModule.from_expr(mod)
		seq = tvm.transform.Sequential(
			[
				relay.transform.InferType(),
				relay.transform.ToMixedPrecision()
			]
		)
		with  tvm.transform.PassContext(opt_level=3):
			mod = seq(mod)
	return  mod

######################################################################
# Let's choose "float16_acc32" for example. 
# You can also use "float16" or "float32" as other dtype options.
dtype="float16_acc32"
mod = convert_to_dtype(mod["main"], dtype)
dtype = "float32"  if  dtype == "float32"  else  "float16"


######################################################################
# Load a test image
# -----------------
# As an example we would use classical cat image from ImageNet

from  PIL  import  Image
from  tvm.contrib.download  import  download_testdata

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path)
img = img.resize(shape_dict[input_name][2:])
img = np.array(img) - np.array([123.0, 117.0, 104.0])
img /= np.array([58.395, 57.12, 57.375])
img = img.transpose((2, 0, 1))
img = img[np.newaxis, :]

######################################################################
# Compile the model with relay
# ----------------------------
# Specify Adreno target before compiling to generate texture 
# leveraging kernels and get all the benefits of textures 

target="opencl -device=adreno"
target_host="llvm -mtriple=arm64-linux-android"

with  tvm.transform.PassContext(opt_level=3):
	lib = relay.build(
		mod, target_host=target_host, target=target, params=params
	)

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# Using RPC you can deploy the model remotely from host
# machine to the remote Adreno device.

rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT"))
key="android"

tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
remote = tracker.request(
	key, priority=0, session_timeout=600000)

temp = utils.tempdir()
dso_binary = "dev_lib_cl.so"
dso_binary_path = temp.relpath(dso_binary)
lib.export_library(dso_binary_path, ndk.create_shared)
remote_path = "/data/local/tmp/" + dso_binary
remote.upload(dso_binary_path)
rlib = remote.load_module(dso_binary)
ctx = remote.cl(0)
m = graph_executor.GraphModule(rlib["default"](ctx))

######################################################################
# Run inference on Adreno
# -----------------------
m.set_input(input_name, tvm.nd.array(img.astype("float32")))
m.run()
tvm_output = m.get_output(0)

######################################################################
# Get predictions and performance statistic
# -----------------------------------------
# This piece of code displays the top-1 and top-5 predictions, as
# well as provides information about the model's performance
from  os.path  import  join, isfile
from  matplotlib  import  pyplot  as  plt
from  tvm.contrib  import  download

# Download ImageNet categories
categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

top_categories = np.argsort(tvm_output.asnumpy()[0])
top5 = np.flip(top_categories, axis=0)[:5]

# Report top-1 classification result
print("Top-1 id: {}, class name: {}".format(top5[1-1], synset[top5[1-1]]))

# Report top-5 classification results
print("\nTop5 predictions: \n")
print("\t#1:", synset[top5[1-1]])
print("\t#2:", synset[top5[2-1]])
print("\t#3:", synset[top5[3-1]])
print("\t#4:", synset[top5[4-1]])
print("\t#5:", synset[top5[5-1]])
print("\t", top5)
ImageNetClassifier = False
for  k  in  top_categories[-5:]:
	if  "cat"  in  synset[k]:
		ImageNetClassifier = True
assert  ImageNetClassifier, "Failed ImageNet classifier validation check"

print("Evaluate inference time cost...")
print(m.benchmark(ctx, number=1, repeat=10))

######################################################################
# Sample Output
# -------------
# The following is the result of 'cpu', 'opencl' and 'vulkan' using Adreno 530 on Snapdragon 820
#
# .. code-block:: bash
#
#    Top-1 id: 282, class name: tiger cat
#
#    Top5 predictions: 
#
#         #1: tiger cat
#         #2: Egyptian cat
#         #3: tabby, tabby cat
#         #4: lynx, catamount
#         #5: red fox, Vulpes vulpes
#          [282 285 281 287 277]
#    Evaluate inference time cost...
#    Execution time summary:
#     mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
#      76.6101      76.6053      77.6530      75.5720       0.6789