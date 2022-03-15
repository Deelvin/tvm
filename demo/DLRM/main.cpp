/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file demo/DLRM/main.cpp
 * \brief native sample for DRM model perfromance and accuracy check.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>
#include <filesystem>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/blockingconcurrentqueue.h>
#include <dmlc/thread_group.h>

#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
// #include <dlpack/dlpack.h>
#include <filesystem>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

struct item {
  std::string name;
  const std::vector<size_t> shape;
  std::string fmt;
};

std::vector<item> s_items = {
  {"p35", {32,2,1,1,256,1,4,16,}, "float32"},
  {"p22", {4,128,}, "float32"},
  {"p40", {1,256,}, "float32"},
  {"p10", {20263,128,}, "float32"},
  {"p0", {32,1,1,1,13,2,1,8,}, "float32"},
  {"p33", {64,1,1,1,479,2,1,8,}, "float32"},
  {"p39", {16,1,1,32,1,16,16,}, "float32"},
  {"p21", {155,128,}, "float32"},
  {"p32", {351,}, "int64"},
  {"p8", {17289,128,}, "float32"},
  {"p1", {1,512,}, "float32"},
  {"p16", {2953546,128,}, "float32"},
  {"p30", {108,128,}, "float32"},
  {"p28", {585935,128,}, "float32"},
  {"p15", {38532951,128,}, "float32"},
  {"p4", {1,16,1,1,32,1,8,8,}, "float32"},
  {"p14", {63,128,}, "float32"},
  {"p19", {2208,128,}, "float32"},
  {"p20", {11938,128,}, "float32"},
  {"p11", {3,128,}, "float32"},
  {"p24", {14,128,}, "float32"},
  {"p27", {39664984,128,}, "float32"},
  {"p37", {16,2,1,1,256,2,4,8,}, "float32"},
  {"p12", {7120,128,}, "float32"},
  {"p41", {1,1,1,128,1,2,1,}, "float32"},
  {"p17", {403346,128,}, "float32"},
  {"p18", {10,128,}, "float32"},
  {"p31", {36,128,}, "float32"},
  {"p26", {25641295,128,}, "float32"},
  {"p29", {12972,128,}, "float32"},
  {"p3", {1,256,}, "float32"},
  {"p23", {976,128,}, "float32"},
  {"p42", {1,}, "float32"},
  {"p2", {16,1,1,32,1,16,16,}, "float32"},
  {"p36", {1,1024,}, "float32"},
  {"p13", {1543,128,}, "float32"},
  {"p9", {7420,128,}, "float32"},
  {"p25", {39979771,128,}, "float32"},
  {"p38", {1,512,}, "float32"},
  {"p5", {1,128,}, "float32"},
  {"p34", {1,1024,}, "float32"},
  {"p6", {39884406,128,}, "float32"},
  {"p7", {39043,128,}, "float32"}
};

// data from day 23
// std::vector<item> s_testData = {
//   {"r_x", {128000, 13}, "float32"},
//   {"lsi", {26, 128000}, "int64"},
//   {"i",   {128000, 1},  "float32"},
// };

// synthetic data
std::vector<item> s_testData = {
  {"batch_dense_X_ref", {2048, 13}, "float32"},
  {"batch_lS_i_ref", {26, 2048}, "int64"},
  {"batch_res_ref",   {2048, 1},  "float32"},
};


typedef std::chrono::high_resolution_clock Clock;

// typedef void (*run_callback)(const std::vector<const DLTensor*> &inputs,
//                             const std::vector<const DLTensor*> &outputs);

void loadNDArray(const std::string& nm,
                 const std::string& pth,
                tvm::runtime::NDArray arr,
                tvm::runtime::PackedFunc set_input)
{
  auto shape = arr.Shape();
  size_t sz = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    sz *= shape[i];
  }
  size_t bytes = arr.DataType().bytes() * sz;
  auto f=fopen((pth + nm).c_str(), "rb");  
  if (f) {
    auto reads = fread((char*)arr->data, 1, bytes,f);
    if (reads != bytes) {
      std::cout << "ERROR: read " << reads << ", but required " << bytes << " bytes from " << nm << " file.\n";
    }
    fclose(f);
  } else {
    std::cout << " File: " << (pth + nm).c_str() << " was not found.\n";
  }
  if (set_input != nullptr) {
    set_input(nm, arr);
  }
}

inline std::vector<size_t> sortedItems(const std::vector<item>& desc) {
  std::vector<size_t> res;
  struct vals {
    size_t sz = 0;
    size_t id = 0;
  };
  std::vector<vals> dims;
  dims.reserve(desc.size());

  size_t idx = 0;
  for (const auto& elem : desc) {
    size_t val = 1;
    for (auto x : elem.shape) {
      val *= x;
    }
    dims.push_back({val, idx});
    ++idx;
  }
  std::sort(dims.begin(), dims.end(), [](const vals& a, const vals& b) {
      return a.sz > b.sz;
  });
  for (auto& elem : dims) {
    res.push_back(elem.id);
  }
  return res;
}

inline void loadData(const std::vector<item>& desc,
              std::vector<tvm::runtime::NDArray>& outData,
              const DLDevice& ctx,
              tvm::runtime::PackedFunc set_input,
              const std::string& pth) {
  outData.resize(desc.size());
  // size_t ind = 0;
  std::vector<std::thread> loadThreads;
  auto indices = sortedItems(desc); // sort to start from "largest" elements
  loadThreads.reserve(desc.size());
  for (auto idx : indices) {
    const auto& el = desc[idx];
    outData[idx] = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple(el.shape.begin(), el.shape.end()),
                                                tvm::runtime::String2DLDataType(el.fmt), ctx);
    loadThreads.emplace_back(std::thread(loadNDArray, el.name, pth, outData[idx], set_input));
    // loadNDArray(el.name, pth, outData[ind], set_input);
    // ind++;
  }
  for (std::thread & thr : loadThreads) {
    if (thr.joinable()) {
      thr.join();
    }
  }
}

template <class T> void fillData(const T* pIn,
                            T* pOut,
                            size_t inOffset,
                            size_t outBatchSize,
                            size_t elemsCout) {
  for (size_t j = 0; j < outBatchSize; ++j) {
    for (size_t i = 0; i < elemsCout; ++i) {
      size_t posOut = j * elemsCout + i;
      size_t posIn = inOffset + posOut;
      pOut[posOut] = pIn[posIn];
    }
  }
}

template <class T> void fillTransposedData(const T* pIn,
                            T* pOut,
                            size_t inOffset,
                            size_t outBatchSize, //batch_size
                            size_t elemsCout, //26
                            size_t num,
                            size_t inBatchSize) {     //batch_size
  for (size_t j = 0; j < num; ++j) {
    for (size_t i = 0; i < elemsCout; ++i) {
      size_t posOut = j + i * outBatchSize;
      size_t posIn = inOffset + j  + i * inBatchSize;
      pOut[posOut] = pIn[posIn];
    }
  }
}

void PrintHelp(char *argv[]) {
  std::filesystem::path p = argv[0];
  std::cout << "command line: " << p.stem().string() << " <reference to model so> <reference to model json> <reference to weights folder> <reference to test data folder>\n";
}

const size_t LOOPS_COUNT = 1000;
const size_t batch_size = 128;
DLDevice ctx = {kDLCPU, 0};

size_t testDataSize = 0;

int test_executor(const std::string& json_data,
                  tvm::runtime::Module mod_lib,
                  tvm::runtime::Module main_mod,
                  const std::vector<tvm::runtime::NDArray>& testData)
{
  std::cout << "get thid " << gettid() << "\n";
  auto callback = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  if (!callback) {
    std::cout << "bad\n";
    return -1;
  }
  std::vector<std::string> names;
  tvm::runtime::Module run_mod =
      (*callback)(json_data, mod_lib, static_cast<int>(ctx.device_type), ctx.device_id);
  tvm::runtime::PackedFunc set_input = run_mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = run_mod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = run_mod.GetFunction("run");
  tvm::runtime::PackedFunc share_params = run_mod.GetFunction("share_params");
  if (share_params != nullptr) {
    share_params(main_mod, "p35", "p22", "p40", "p10",
                           "p0",  "p33", "p39", "p21",
                          "p32",  "p8",  "p1",  "p16",
                          "p30", "p28",  "p15",  "p4",
                          "p14",  "p19",  "p20",  "p11",
                          "p24",  "p27",  "p37",  "p12",
                          "p41",  "p17",  "p18",  "p31",
                          "p26",  "p29",  "p3",  "p23",
                          "p42",  "p2",  "p36",  "p13",
                          "p9",  "p25",  "p38",  "p5",
                          "p34",  "p6",  "p7");
  }
  tvm::runtime::NDArray input_i =
  tvm::runtime::NDArray::Empty({batch_size, 13}, tvm::runtime::String2DLDataType("float32"), ctx);
  tvm::runtime::NDArray ls_o =
  tvm::runtime::NDArray::Empty({26, batch_size}, tvm::runtime::String2DLDataType("int64"), ctx);
  tvm::runtime::NDArray ls_i =
    tvm::runtime::NDArray::Empty({26, batch_size}, tvm::runtime::String2DLDataType("int64"), ctx);

  tvm::runtime::NDArray output =
    tvm::runtime::NDArray::Empty({batch_size}, tvm::runtime::String2DLDataType("float32"), ctx);

  if (run != nullptr) {
    for (size_t i = 0; i < 5; ++i)
      run();
  }
  size_t testDataSize = 0;
  float* pInput_In = (float*)testData[0]->data;
  int64_t* pls_i_In = (int64_t*)testData[1]->data;
  testDataSize = testData[2].Shape()[0];
  int64_t time = 0;
  float* pInput = (float*)input_i->data;
  int64_t* pls_o = (int64_t*)ls_o->data;
  int64_t* pls_i = (int64_t*)ls_i->data;
  std::vector<float> results(testDataSize);
  for (size_t cnt = 0; cnt < LOOPS_COUNT; ++cnt) {
    size_t runsCount = 0;
    for (size_t pos = 0; pos < testDataSize; pos += batch_size) {
      size_t num = testDataSize - pos;
      if (num < batch_size > testDataSize) {
        std::memset(pInput, 0, batch_size * 13 * sizeof(float));
        std::memset(pls_i, 0, batch_size *  26 * sizeof(int64_t));
        fillData(pInput_In, pInput, pos * 13, num, 13);
        fillTransposedData(pls_i_In, pls_i, pos, batch_size, 26, num, testDataSize);
      } else {
        fillData(pInput_In, pInput, pos * 13, batch_size, 13);
        fillTransposedData(pls_i_In, pls_i, pos , batch_size, 26, batch_size, testDataSize);
      }
      set_input("input.1", input_i);
      set_input("lS_o",    ls_o);
      set_input("lS_i",    ls_i);

      auto ts_start = Clock::now();
      run();
      auto ts_end = Clock::now();
      time += std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_start).count();
      tvm::runtime::NDArray res = get_output(0);
      float* pRes = (float*)res->data;
      
      for (size_t i = 0; i < std::min(batch_size, num); ++i) {
        results[runsCount * batch_size + i] = pRes[i];
      }
      runsCount++;
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  const char* env = getenv("TVM_NUM_THREADS");
  if (argc != 5) {
    PrintHelp(argv);
    return -1;
  }
  size_t threads_count;
  if (env) {
    std::stringstream sstream(env);
    
    sstream >> threads_count;
    
  }
  std::cout << "current threads = " << threads_count << "\n";
  auto so_path = argv[1];
  std::string s_tmpdir = argv[3];
  s_tmpdir += "/";
  std::string s_testDataPath = argv[4];
  s_testDataPath += "/";

  if (!std::filesystem::exists(so_path)) {
    std::cout << "ERROR: model library file was not found: " << so_path << "\n";
    return -1;
  }

  const std::string json_file(argv[2]);
  tvm::runtime::Module mod_lib = tvm::runtime::Module::LoadFromFile(so_path);
  std::ifstream json_in(json_file.c_str(), std::ios::in);
  if (!json_in.is_open()) {
    std::cout << "ERROR: model json file was not found: " << json_file << "\n";
    return -1;
  }
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

int main(int argc, char *argv[]) {
  const char* env = getenv("TVM_NUM_THREADS");
  if (argc != 5) {
    PrintHelp(argv);
    return -1;
  }
  if (env) {
    std::stringstream sstream(env);
    size_t v;
    sstream >> v;
    std::cout << "current threads = " << v << "\n";
  }

  auto so_path = argv[1];
  std::string s_tmpdir = argv[3];
  s_tmpdir += "/";
  std::string s_testDataPath = argv[4];
  s_testDataPath += "/";

  if (!std::filesystem::exists(so_path)) {
    std::cout << "ERROR: model library file was not found: " << so_path << "\n";
    return -1;
  }

  const size_t LOOPS_COUNT = 50;
  const size_t batch_size = 128;
  DLDevice ctx = {kDLCPU, 0};

  const std::string json_file(argv[2]);
  tvm::runtime::Module mod_lib = tvm::runtime::Module::LoadFromFile(so_path);
  std::ifstream json_in(json_file.c_str(), std::ios::in);
  if (!json_in.is_open()) {
    std::cout << "ERROR: model json file was not found: " << json_file << "\n";
    return -1;
  }
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();
  std::string params_data = "";
  constexpr int device_type = kDLCPU;
  constexpr int device_id = 0;
  auto callback = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  if (!callback) {
    std::cout << "bad\n";
    return -1;
  }
  tvm::runtime::Module run_mod =
      (*callback)(json_data, mod_lib, static_cast<int>(device_type), device_id);
  tvm::runtime::PackedFunc set_input = run_mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = run_mod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = run_mod.GetFunction("run");
  std::vector<tvm::runtime::NDArray> weights;
  std::vector<tvm::runtime::NDArray> testData;
  loadData(s_items, weights, ctx, set_input, s_tmpdir);
  loadData(s_testData, testData, ctx, nullptr, s_testDataPath);

  tvm::runtime::NDArray input_i =
    tvm::runtime::NDArray::Empty({batch_size, 13}, tvm::runtime::String2DLDataType("float32"), ctx);
  tvm::runtime::NDArray ls_o =
    tvm::runtime::NDArray::Empty({26, batch_size}, tvm::runtime::String2DLDataType("int64"), ctx);
  tvm::runtime::NDArray ls_i =
    tvm::runtime::NDArray::Empty({26, batch_size}, tvm::runtime::String2DLDataType("int64"), ctx);

  tvm::runtime::NDArray output =
    tvm::runtime::NDArray::Empty({batch_size}, tvm::runtime::String2DLDataType("float32"), ctx);

  testDataSize = testData[2].Shape()[0];
  int64_t time = 0;
  float* pInput = (float*)input_i->data;
  int64_t* pls_o = (int64_t*)ls_o->data;
  int64_t* pls_i = (int64_t*)ls_i->data;

  float* pInput_In = (float*)testData[0]->data;
  // int64_t* pls_o_In = (int64_t*)testData[1]->data;
  int64_t* pls_i_In = (int64_t*)testData[1]->data;
  size_t totalCount = 0;
  std::vector<float> results(testDataSize);
  for (size_t i = 0; i < 26; ++i) {
      for (size_t j = 0; j < batch_size; ++j) {
        pls_o[i * batch_size + j ] = j;
      }
  }
  TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
  size_t testDataSize = 0;
  std::shared_ptr<dmlc::ThreadGroup> thread_group = std::make_shared<dmlc::ThreadGroup>();
  for(int x = 1; x < threads_count; ++x) {
    dmlc::ThreadGroup::Thread::SharedPtr thread =
      std::make_shared<dmlc::ThreadGroup::Thread>(std::string("test_thread_ar ")
                                                         + std::to_string(x), thread_group.get());
    dmlc::ThreadGroup::Thread::launch(thread, true, test_executor, json_data, mod_lib, run_mod, testData);
  }
  // warm-up
  for (size_t i = 1; i < 5; ++i)
    run();

  for (size_t cnt = 0; cnt < LOOPS_COUNT; ++cnt) {
    size_t runsCount = 0;
    for (size_t pos = 0; pos < testDataSize; pos += batch_size) {
      size_t num = testDataSize - pos;
      if (num < batch_size > testDataSize) {
        std::memset(pInput, 0, batch_size * 13 * sizeof(float));
        std::memset(pls_i, 0, batch_size *  26 * sizeof(int64_t));
        fillData(pInput_In, pInput, pos * 13, num, 13);
        fillTransposedData(pls_i_In, pls_i, pos, batch_size, 26, num, testDataSize);
      } else {
        fillData(pInput_In, pInput, pos * 13, batch_size, 13);
        fillTransposedData(pls_i_In, pls_i, pos , batch_size, 26, batch_size, testDataSize);
      }
      set_input("input.1", input_i);
      set_input("lS_o",    ls_o);
      set_input("lS_i",    ls_i);

      auto ts_start = Clock::now();
      run();
      auto ts_end = Clock::now();
      time += std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_start).count();
      ++totalCount;
      tvm::runtime::NDArray res = get_output(0);
      float* pRes = (float*)res->data;
      
      for (size_t i = 0; i < std::min(batch_size, num); ++i) {
        results[runsCount * batch_size + i] = pRes[i];
      }
      runsCount++;
    }
  }
  thread_group.reset();
  std::cout << time / totalCount << " us." << std::endl;
  float* pRef = (float*)testData[2]->data;
  // accuracy check
  size_t equal = 0;
  float error = 0;
  float maxError = 0;
  for (size_t i = 0; i < testDataSize; ++i) {
    // if ((i %= 128) == 0)
    //   std::cout << "---\n";
    // std::cout << pRef[i]  << " " << results[i] << "\n";
    auto currErr = std::fabs(pRef[i] - results[i]);
    error += currErr;
    maxError = std::max(currErr, maxError);
    if (pRef[i] == std::round(results[i])) {
      equal++;
    }
  }
  return 0;
}
