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
#include <atomic>
#include <stdlib.h>

#include "workload.hpp"
#include "os_tools.hpp"


void check_freq_(const int num_workers, bool ht) {
  std::atomic<bool> continue_work {true};
  const int benchmark_time_ms = 1000; // 1 sec
  std::vector<std::thread> workers(num_workers);
  std::vector<double> max_worker_ips(num_workers);
  
  auto workload = [&continue_work] (int idx, double *res) {
    pin_thread(idx);
    double max_ipc = 0.0;
    while (continue_work.load()) {
      auto ipc = avx512Mandel(0.1, 0.05, 5);
      max_ipc = std::max(max_ipc, ipc); 
    }
    *res = max_ipc;
  };

  const int idx_scale = ht ? 2 : 1;
  int idx = 0;
  for (auto &worker : workers) {
    worker = std::thread(workload, idx * idx_scale, &max_worker_ips[idx]);
    idx++;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(benchmark_time_ms/2));
  auto cur_freq = get_cur_freq();
  std::this_thread::sleep_for(std::chrono::milliseconds(benchmark_time_ms/2));

  // Finish processing
  continue_work.store(false);
  for (auto & w : workers)
    w.join();

  std::vector<float> worker_freq;
  std::vector<float> idel_freq;
  for (size_t i = 0; i < cur_freq.size(); i++)
    if (i % idx_scale == 0 && i / idx_scale < num_workers)
      worker_freq.push_back(cur_freq[i]);
    else 
      idel_freq.push_back(cur_freq[i]);

  auto worker_stat = calc_statistic_4m(worker_freq);
  auto idel_stat = calc_statistic_4m(idel_freq);

  const bool hunam_readable = true;
  if (hunam_readable) {
    std::cout << "Workers #" << num_workers << std::endl;
    std::cout << "   workers : " << worker_stat.min << "-" << worker_stat.max 
              << " med " << worker_stat.median << " GHz  " 
              << " mean " << worker_stat.mean << " GHz" 
              << std::endl;
    std::cout << "      idel : " << idel_stat.min << "-" << idel_stat.max 
              << " med " << idel_stat.median << " GHz " 
              << " mean " << idel_stat.mean << " GHz" 
              << std::endl;
    
    for (auto ipc : max_worker_ips) std::cout << ipc << " "; 
    std::cout << std::endl;
  } else {
    std::cout << num_workers << " " 
              << worker_stat.min << " "
              << worker_stat.max << " "
              << max_worker_ips[0] << " "
              << max_worker_ips[max_worker_ips.size() - 1] << " "
              << std::endl;
  }
}

void check_freq() {
  auto info = get_system_info();
  for (int num = 1; num < info.num_of_cores + 1; num++)
    check_freq_(num, info.hyper_threading);
}

void check_madd_throughput() {
  auto ipc = avx512Mandel(0.1, 0.05, 20);
  std::cout << "AVG throughput of mul : " << ipc << std::endl << std::endl;
  if (0.5 < ipc && ipc < 1.0)
    std::cout << "Looks like system has 2 AVX512 FMA units" << std::endl;
  else if (1.0 < ipc && ipc < 1.5)
    std::cout << "Looks like system has 1 AVX512 FMA units" << std::endl;
  else
    std::cout << "No conclusionsabount num of AVX512 FMA units" << std::endl;
}

int main() {
  std::cout << "DNNL benchmark tool" << std::endl << std::endl;  
  
  print_system_info();
  
  check_madd_throughput();

  check_freq();
}