# amalyshe - I added here version 2.8.12 just because it was used for DLDT, no other reasons so far
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 17)

project(dlrm_infer)

set(TVM_HOME $ENV{TVM_HOME})
if(NOT DEFINED TVM_HOME)
        message(FATAL_ERROR " Cannot find path to TVM, please set TVM_HOME environment variable")
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
