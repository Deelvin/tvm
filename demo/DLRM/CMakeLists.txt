cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 17)

project(dlrm_infer)

set(TVM_HOME $ENV{TVM_HOME})
message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT DEFINED TVM_HOME)
  set(TVM_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  message(" Cannot find path to TVM_HOME , the current path is used as reference: ${TVM_HOME}")
else()
  message("Found TVM at ${TVM_HOME}")
endif()

add_executable(dlrm_infer
               main.cpp
)

target_include_directories(dlrm_infer PRIVATE
                                        "${TVM_HOME}/include"
                                        "${TVM_HOME}/3rdparty/dmlc-core/include/"
                                        "${TVM_HOME}/3rdparty/dlpack/include"
                                        )


target_link_libraries(dlrm_infer "${TVM_HOME}/build/libtvm_runtime.so"
                       pthread
                      )
