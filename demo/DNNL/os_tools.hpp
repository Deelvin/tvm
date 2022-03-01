#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <sched.h>

struct value_statistic {
  float median;
  float min;
  float max;
  float mean;
};

template<typename T>
value_statistic calc_statistic_4m(std::vector<T> arr) {
  value_statistic res;
  std::sort(arr.begin(), arr.end());
  res.median = arr[arr.size() / 2];
  res.min = arr[0];
  res.max = arr[arr.size() - 1];
  res.mean = 0.0f;
  for (const auto el : arr) res.mean += el;
  res.mean /= arr.size();

  return res;
}


std::vector<std::vector<std::string>> grep_cpu_info(const std::vector<std::string> &tags) {
  std::ifstream cpuinfo("/proc/cpuinfo");
  
  std::vector<std::vector<std::string>> values(tags.size());
  std::string line;
  while (std::getline(cpuinfo, line)) {
    for (size_t i = 0; i < tags.size(); i++) { 
      if (!line.compare(0, tags[i].size(), tags[i])) {
        static const std::string del = ": ";
        auto pos = line.find(del);
        values[i].push_back(line.substr(pos + del.size()));
      }
    }
  }
  return  values;
}

struct sys_info {
    std::string name;
    float base_freq_ghz;
    int num_of_cores;
    int num_of_socets;
    bool hyper_threading;
    bool avx2;
    bool avx512;
};

bool hyper_threading_is_used() {
  auto f_name = "/sys/devices/system/cpu/smt/active";
  std::ifstream cpuinfo(f_name);
  std::string line;
  std::getline(cpuinfo, line);
  return std::stoi(line) == 1; 
}

sys_info construct_system_info_() {
    auto fields = grep_cpu_info({"processor", "model name", "cpu MHz", "flags", "cpu cores"});
    auto indexes = fields[0];
    auto names = fields[1];
    auto freqs = fields[2];
    auto flags = fields[3];
    auto cores_per_socket = std::stoi(fields[4][0]);
    
    // Assumption! Median value of cur freq is a base freq of CPU.
    // In idle state most of cores stay with base frequancy.  
    auto freqs_f = std::vector<float>(freqs.size());
    std::transform(freqs.begin(), freqs.end(), freqs_f.begin(), [] (const std::string &val) { return std::stof(val); });
    std::sort(freqs_f.begin(), freqs_f.end());
    auto base_freq_ghz = freqs_f[freqs_f.size()/2];
    base_freq_ghz /= 1000; // move to GHz

    bool avx512 = (flags[0].find("avx512f") != std::string::npos); 
    bool avx2 = (flags[0].find("avx2") != std::string::npos); 

    sys_info res;
    res.name = fields[1][0];
    res.base_freq_ghz = base_freq_ghz;
    res.num_of_cores = indexes.size();
    res.avx2 = avx2;
    res.avx512 = avx512;

    res.hyper_threading = hyper_threading_is_used();
    if (res.hyper_threading)
      res.num_of_cores /= 2;

    res.num_of_socets = res.num_of_cores / cores_per_socket;
    return res;
}

sys_info get_system_info() {
  static const sys_info system_info = construct_system_info_();
  return system_info;
}

float calc_theoretical_peak(float boost_freq = 0.0f) {
  auto info = get_system_info();
  double base_freq = boost_freq == 0.0f ? info.base_freq_ghz : boost_freq;
  double simd_size = info.avx512 ? 16 : info.avx2 ? 8 : 1;   
  double ipc = 2;  // throughput of madd instruction. Equal 2 for most of modern Intel/AMD CPUs. 
  return base_freq * simd_size * ipc; // GFLOPS
}

void print_system_info() {
    auto system_info = get_system_info();
    auto peak_gflops = calc_theoretical_peak(); 

    std::cout << "==================================" << std::endl;
    std::cout << " Name   : " << system_info.name << std::endl;
    std::cout << " Freq   : " << system_info.base_freq_ghz << " GHz" << std::endl;
    std::cout << " Cores  : " << system_info.num_of_cores << std::endl;
    std::cout << " SIMD   : " << (system_info.avx2 ? "AVX2 " : " ") << (system_info.avx512 ? "AVX512 " : " ") << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << " Single core PEAK  : " << peak_gflops << " GFLOPS" << std::endl;
    std::cout << " Multi core PEAK   : " << peak_gflops * system_info.num_of_cores << " GFLOPS" << std::endl;
    std::cout << "==================================" << std::endl;
}

std::vector<float> get_cur_freq() {
    auto fields = grep_cpu_info({"cpu MHz"});
    auto freqs = fields[0];
    
    std::vector<float> res(freqs.size());
    for (size_t i = 0; i < freqs.size(); i++) {
        res[i] = std::stof(freqs[i]) / 1000; 
    }
    return res;
}

std::vector<float> get_cur_freq_2() {
    auto system_info = get_system_info();
    std::vector<float> res(system_info.num_of_cores);

    for (int i = 0; i < system_info.num_of_cores; i++) {
        auto f_name = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq";
        std::ifstream cpuinfo(f_name);
        std::string line;
        std::getline(cpuinfo, line);
        res[i] = std::stof(line) / 1000; 
    }

    return res;
}

int get_num_cores() {
  auto system_info = get_system_info();
  return system_info.num_of_cores;
}

void pin_thread(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
