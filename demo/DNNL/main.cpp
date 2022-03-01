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
 * \file demo/DNNL/main.cpp
 * \brief Naive benchmark application
 */

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <thread>
#include <atomic>
#include <map>
#include <stdlib.h>

#include <dnnl.hpp>
#include <omp.h>

#include "workload.hpp"
#include "os_tools.hpp"

struct PrimSpec {
  dnnl::primitive::kind prim_kind;
  union {
    struct {
      int N;
      int IC;
      int OC;
      dnnl::algorithm act;
    } dense;

    struct {
      int N;
      int IC, IH, IW;
      int OC, KH, KW;
      int pad, stride, dilation;
      dnnl::algorithm act;
    } conv;
  };

  dnnl::memory::data_type dtype_src, dtype_wgh, dtype_dst;
};

PrimSpec parse_prim_spec(std::string ser_spec) {
  // TBD
  return {};
};

std::vector<PrimSpec> parse_workloads(std::istream in) {
  std::vector<PrimSpec> res;
  std::string line;
  while (std::getline(in, line)) {
    auto spec = parse_prim_spec(line);
    res.push_back(spec);
  }

  return res;
}

void fill_random(dnnl::memory mem) {
  if (mem.get_desc().data_type() == dnnl::memory::data_type::f32) {
    auto size = mem.get_desc().get_size() / sizeof(float);
    auto ptr = static_cast<float*>(mem.get_data_handle());

    std::mt19937 gen;
    std::uniform_real_distribution<float> rand_gen;

    for (size_t i = 0; i < size; i++)
      ptr[i] = rand_gen(gen);

  } else {
    // TBD
    throw std::invalid_argument("Unsupported dtype");
  }
}

struct SUT {
  SUT(std::function<void()> action_, long macs_) : action(action_), macs(macs_) {}
  SUT() : action(), macs(0) {}
  void operator()() const { action(); }
  
  std::function<void()> action;
  std::function<void()> initialize;
  long macs;
};

SUT prepareSUT(PrimSpec spec) {
  if (spec.prim_kind == dnnl::primitive::kind::convolution) {
    // TBD
  } else if (spec.prim_kind == dnnl::primitive::kind::inner_product) {
    auto N = spec.dense.N;
    auto IC = spec.dense.IC;
    auto OC = spec.dense.OC;

    auto src_desc = dnnl::memory::desc({N, IC}, spec.dtype_src, dnnl::memory::format_tag::any);
    auto dst_desc = dnnl::memory::desc({N, OC}, spec.dtype_src, dnnl::memory::format_tag::any);
    auto wgh_desc = dnnl::memory::desc({OC, IC}, spec.dtype_wgh, dnnl::memory::format_tag::any);
    auto bis_desc = dnnl::memory::desc({OC}, spec.dtype_src, dnnl::memory::format_tag::any);

    auto desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference,
                                                  src_desc, wgh_desc, bis_desc, dst_desc);

    auto eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
    auto p_desc = dnnl::inner_product_forward::primitive_desc(desc, eng);
    auto prim = dnnl::inner_product_forward(p_desc);

    auto src = dnnl::memory(p_desc.src_desc(), eng);
    auto dst = dnnl::memory(p_desc.dst_desc(), eng);
    auto wgh = dnnl::memory(p_desc.weights_desc(), eng);
    auto bis = dnnl::memory(p_desc.bias_desc(), eng);

    fill_random(src);
    fill_random(dst);
    fill_random(wgh);
    fill_random(bis);

    dnnl::stream strm(eng);
    std::unordered_map<int, dnnl::memory> args {
      {DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wgh}, {DNNL_ARG_BIAS, bis}, {DNNL_ARG_DST, dst}
    };

    return {[prim, strm, args] () {
      prim.execute(strm, args);
    }, N*IC*OC};
  }
  return {};
}

SUT prepareSUT_TVM(std::string path) {
  DLDevice dev{kDLCPU, 0};
  auto mod_factory = tvm::runtime::Module::LoadFromFile(path);
  // create the graph runtime module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

  return {[run] () { run(); }, 100500};
}

int set_affinity_action(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  return 0;
}


void set_affinity(int arena_size, int arena_id) {
  auto system_info = get_system_info();
  auto core_idx_multiplier = system_info.hyper_threading ? 2 : 1; // ignore HT cores 
  #pragma omp parallel num_threads(arena_size)
  {
      auto thr_idx = omp_get_thread_num();
      auto core_id = thr_idx + arena_size * arena_id * core_idx_multiplier;
      set_affinity_action(core_id);
  }
}

std::array<value_statistic, 2> benchmark(const SUT &sut, int num_thr = 1, int num_inst = 1, int benchmark_time_ms = 200 ) {
  using Time = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;

  std::atomic<bool> loop_cond;
  loop_cond.store(true);

  auto worker_loop = [&loop_cond, sut, num_thr] (int idx, long &res_count, long &res_dur_us) {
    int count = 0;
    set_affinity(num_thr, idx);
    auto start = Time::now();
    while (loop_cond.load()) {
      sut();
      count++;
    }
    res_dur_us = std::chrono::duration_cast<us>(Time::now() - start).count();
    res_count = count;
  };

  std::vector<std::thread> workers(num_inst);
  std::vector<long> counts(num_inst);
  std::vector<long> durs(num_inst);
  // Start worker threads
  for (size_t i = 0; i < workers.size(); i++)
     workers[i] = std::thread(worker_loop, i, std::ref(counts[i]), std::ref(durs[i]));

  // Wait for half of benchmarking time
  std::this_thread::sleep_for(std::chrono::milliseconds(benchmark_time_ms/2));
  auto actual_freq = get_cur_freq();
  actual_freq.resize(num_thr * num_inst);
  std::this_thread::sleep_for(std::chrono::milliseconds(benchmark_time_ms/2));

  // Finish processing
  loop_cond.store(false);
  for (auto & w : workers)
    w.join();

  std::vector<float> latensies(num_inst);
  for (int i = 0; i < num_inst; i++) {
    latensies[i] = float(durs[i]) / counts[i]; // time per sample
  }

  
  auto freq_stat = calc_statistic_4m(actual_freq);
  auto ltns_stat = calc_statistic_4m(latensies);

  return {ltns_stat, freq_stat};
} 

void benchmark_sut(const SUT &sut, int num_threads = 1, int num_instance = 1) {
  omp_set_num_threads(num_threads);

  auto score = benchmark(sut, num_threads, num_instance, /* dur= */ 10000);
  auto ltns = score[0];
  auto freq = score[1];

  const auto macs = sut.macs;
  const auto throughput_per_core_gflops = static_cast<int>(float(macs) / ltns.mean / 1000 / num_threads);
  const auto throughput_total_gflops = static_cast<int>(float(macs) / ltns.mean / 1000 * num_instance);
  const auto boost_peak = calc_theoretical_peak(freq.mean);
  const auto effitiency = throughput_per_core_gflops / boost_peak;

  std::cout << "==================================" << std::endl;
  std::cout << "INST: " << num_instance << " THRD: " << num_threads << std::endl;
  std::cout << "LTNS: " << ltns.mean << " FREQ: " << freq.mean << std::endl;
  std::cout << "THRP_CORE: " << throughput_per_core_gflops << " THRP_TOT: " << throughput_total_gflops 
            << " EFFIC: " << effitiency << std::endl;
}

int main() {
  std::cout << "DNNL benchmark tool" << std::endl;
  
  print_system_info();

  int n_thr = 13;
  //XXX  before dnnl calls
  omp_set_num_threads(n_thr);
  //XXX
  
  PrimSpec spec;
  spec.prim_kind = dnnl::primitive::kind::inner_product;
  spec.dense.N = 100;
  spec.dense.IC = 1024;
  spec.dense.OC = 1024;
  spec.dtype_src = dnnl::memory::data_type::f32;
  spec.dtype_dst = dnnl::memory::data_type::f32;
  spec.dtype_wgh = dnnl::memory::data_type::f32;
  auto sut = prepareSUT(spec);

  for (int n_inst = 1; n_inst < 3; n_inst++) {
    benchmark_sut(sut, n_thr, n_inst);
  }
}