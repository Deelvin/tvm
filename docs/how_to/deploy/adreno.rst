Compile and deploy models on Adreno GPU
=======================================

**Authors**: Daniil Barinov, Egor Churaev, Andrey Malyshev

Introduction
------------

Adreno is a series of graphicfdgs processing unit (GPU) semiconductor
intellectual property cores developed by Qualcomm and used in many of
their SoCs.

The Adreno GPU accelerates the rendering of complex geometries to
deliver high-performance graphics and a rich user experience with low
power consumption

This guide will demonstrate the benefits of using textures with Adreno,
how to build TVM with OpenCL (needed by Adreno devices) and TVM RPC
enabled. It will also provide example code to compile and deploy models
on Adreno devices.

Advantages of the Textures
--------------------------

One of the advantages of Adreno is its clever handling of textures. At
the moment, TVM is able to benefit from this by having texture support
for Adreno. The graph below shows the Adreno A5x architecture.

|High-level overview of the Adreno A5x architecture for OpenCL| Fig. 1
High-level overview of the Adreno A5x architecture for OpenCL

Reasons of using textures:

-  TP has a dedicated L1 cache, which is read-only cache and stores data
   fetched from level-2 (L2) cache for texture operations (primary
   reason)

-  Handling of image boundaries is built-in.

-  Supports numerous image format and data type combinations with
   support for automatic format conversions

Multiple texturing or multitexturing is the use of more than one texture
at a time on a polygon. Adreno GPUs support up to 32 total textures in a
single render pass, i.e., up to 16 textures in the fragment shader and
up to 16 textures at a time for the vertex shader.

Effective use of multiple textures significantly reduces overdraw, saves
ALU cost for fragment shaders, and avoids unnecessary vertex transforms.

With textures, it is possible to achieve a significant performance boost
compared to OpenCL buffer based solutions.

Building TVM for Adreno
-----------------------

This section gives instructions on how to build the Android part of TVM
via OpenCL-SDK and TVM RPC Server in order to deploy models on Adreno.

Since the process of building TVM for Adreno is exactly the same as the
process of building TVM for Android, please refer to these instructions:
`TVM RPC
Server <https://github.com/apache/tvm/tree/main/apps/cpp_rpc>`__
Alternatively, to build a TVM via docker using OpenCL-Headers and set-up
with Android TVM RPC, refer to this guide: `Deploy the Pretrained Model
on
Android <https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_android.html>`__

**Prerequisites**: Android NDK, Android Debug Bridge (adb), OpenCL

For us to begin with, Android NDK, Android Debug Bridge and OpenCL must
be installed and Android part of TVM must be builded.

**Android NDK installation**: https://developer.android.com/ndk

**Android Debug Bridge installation**:
https://developer.android.com/studio/command-line/adb

**OpenCL installation**: https://github.com/KhronosGroup/OpenCL-SDK.git

You can also build the android part of TVM via console. From the root
folder of TVM:

::

   mkdir build_android
   cd build_android
   cmake .. -DUSE_OPENCL=path/to/OpenCL -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=android-28 -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON -DANDROID_STL=c++_static -DUSE_CPP_RPC=ON
   make -jN tvm_runtime tvm_rpc

where **N** is the number of cores available on your *CPU*.

At this stage you have builded TVM for Adreno.

Build and deploy model for Adreno
---------------------------------

As the deployment pipelines on Android and on Adreno do not have many
differences, in this section we will focus on one and only necessary
aspect to compile and deploy models for Adreno and will take a look at
the generation of kernels with and without textures. In addition, the
possibility of choosing a different precision for model compilation will
be considered.

| |Android deployment pipeline|
| Fig.2 Deployment pipeline on Adreno devices ### Adreno target
  Normally, when compiling models on Android using OpenCL, the
  corresponding target is used

.. code:: python

   target="opencl"

Using Adreno, we want to get all the benefits of textures, so we have to
use the following target to generate texture leveraging kernels

.. code:: python

   target="opencl -device=adreno"

Let’s write simple model and take a look at generated kernels for these
two targets

.. code:: python

   import tvm
   from tvm import relay
   import numpy as np

   input_shape=(1, 56, 56, 32)
   filter_shape=(3, 3, 32, 64)
   filter = np.random.rand(*filter_shape)

   dtype="float32"
   input = tvm.relay.var("input", shape=input_shape, dtype=dtype)
   weight = tvm.relay.var("weight", shape=filter_shape, dtype=dtype)
   D = relay.nn.conv2d(input, weight, padding=(1, 1), data_layout="NHWC", kernel_layout="HWIO", out_dtype=dtype)

   mod = relay.Function([input, weight], D)
   params = {
   "weight": tvm.nd.array(filter)
   }

Classical opencl target:

.. code:: python

   target="opencl"

   with tvm.transform.PassContext(opt_level=3):
   graph, lib, params = relay.build_module.build(mod, target, params=params)
   print(lib.imported_modules[0].get_source())

Notice the generated convolution kernel and the presence of pointers in
the initialization of the function:

.. code:: c

   __kernel void tvmgen_default_fused_nn_conv2d_kernel0(__global float* restrict p0, __global double* restrict p1, __global float* restrict conv2d_nhwc) {
   // body..

The kernels generated with the above target are buffer-based.

Let’s take a look at “opencl -device=adreno” target:

.. code:: python

   target="opencl -device=adreno"

   with tvm.transform.PassContext(opt_level=3):
   graph, lib, params = relay.build_module.build(mod, target, params=params)
   print(lib.imported_modules[0].get_source())

We can now observe the use of textures in the initialization of the
function:

.. code:: c

   __kernel void tvmgen_default_fused_nn_conv2d_kernel0(__write_only image2d_t pad_temp_global_texture, __read_only image2d_t p0) {
   // body..

Precisions
~~~~~~~~~~

We can also set different precision, choosing from *float16*,
*float16_acc32* (Mixed Precision), *float32*. First of all, we need to
register conversion to mixed precision

.. code:: python

   from  tvm.relay.op  import  register_mixed_precision_conversion

   conv2d_acc = "float32"

   # Pick a priority > 10 to overwrite defaults, higher priorities take precedence
   @register_mixed_precision_conversion("nn.conv2d", level=11)
   def  conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
       global  conv2d_acc
       return [
           # always do main calculation in mixed_precision_type
           relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
           # the dtype for the accumulator
           conv2d_acc,
           # the output dtype for the operation (usually fp16)
           mixed_precision_type,
       ]

   # Same for dense
   @register_mixed_precision_conversion("nn.dense", level=11)
   def  conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
       global  conv2d_acc
       return [
           relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
           conv2d_acc,
           mixed_precision_type,
       ]

We then need to obtain **mod**, after which we can convert it to
required **dtype** and then assemble our model sequentialy

.. code:: python

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
                   relay.transform.ToMixedPrecision() # primary method
               ]
           )
           with  tvm.transform.PassContext(opt_level=3):
               mod = seq(mod)
       return  mod

   dtype="float16_acc32"
   mod = convert_to_dtype(mod["main"], dtype)
   dtype = "float32"  if  dtype == "float32"  else  "float16"

From this point we can compile our model as normal

.. code:: python

   with  tvm.transform.PassContext(opt_level=3):
       lib = relay.build(
           mod, target_host=target_host, target=target, params=params
       )

The complete step-py-step process of compiling and deploying models on
Adreno, including selection of precision, running the inference of the
model, getting the predictions, and measuring the performance can be
found in this tutorial: `How To <>`__

.. |High-level overview of the Adreno A5x architecture for OpenCL| image:: https://i.ibb.co/yXm6CkG/2022-10-21-14-39-08.png
.. |Android deployment pipeline| image:: https://i.ibb.co/xMQrgLn/Untitled-Frame-2.jpg
