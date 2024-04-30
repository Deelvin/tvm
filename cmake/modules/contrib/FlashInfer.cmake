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

if(USE_FLASHINFER)
  set(FLASHINFER_GEN_COND "$<BOOL:${USE_CUDA}>")

  tvm_file_glob(GLOB FLASHINFER_RUNTIME_SRCS
    src/runtime/contrib/flashinfer/*.cu
  )

  add_library(flashinfer_runtime_objs OBJECT ${FLASHINFER_RUNTIME_SRCS})
  target_include_directories(flashinfer_runtime_objs PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/flashinfer/include
  )

  list(APPEND FLASHINFER_RUNTIME_OBJS "$<${FLASHINFER_GEN_COND}:$<TARGET_OBJECTS:flashinfer_runtime_objs>>")
  list(APPEND TVM_RUNTIME_EXT_OBJS "${FLASHINFER_RUNTIME_OBJS}")
endif()
