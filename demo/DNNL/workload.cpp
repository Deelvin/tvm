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
 * \file demo/DNNL/workload.hpp
 * \brief Simple kernels avx2 and avx512
 *
 * Original link:
 * https://www.intel.com/content/www/us/en/develop/articles/
 * accelerating-compute-intensive-workloads-with-intel-avx-512-using-microsoft-visual-studio.html
 */

#include "workload.hpp"

#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <stdint.h>

uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

#ifdef __AVX512F__
/* AVX512 Implementation */
__m512i avx512Mandel_(__m512 c_re16, __m512 c_im16, uint32_t max_iterations) {
  constexpr int NUM=8;
  __m512 z_re16[NUM];
  __m512 z_im16[NUM];
  __m512 four16 = _mm512_set1_ps(4.0f);
  __m512 two16 = _mm512_set1_ps(2.0001f);
  __m512i one16 = _mm512_set1_epi32(1);
  __m512i result[NUM];

  for (int i = 0; i < NUM; i++) {
    z_re16[i] = c_re16;
    z_im16[i] = c_im16;
    result[i] = _mm512_setzero_si512();
  }

  // #pragma clang loop unroll(enable)
  // #pragma GCC unroll 2
  for (size_t i = 0; i < max_iterations; i++) {
    // #pragma clang loop unroll(enable)
    // #pragma GCC unroll NUM
    for (int n = 0; n < NUM; n++) {
      __m512 z_im16sq = _mm512_mul_ps(z_im16[n], z_im16[n]);
      __m512 z_re16sq = _mm512_mul_ps(z_re16[n], z_re16[n]);
      __m512 new_im16 = _mm512_mul_ps(z_re16[n], z_im16[n]);
      // __m512 z_abs16sq = _mm512_add_ps(z_re16sq, z_im16sq);
      __m512 new_re16 = _mm512_sub_ps(z_re16sq, z_im16sq);
      // __mmask16 mask = _mm512_cmp_ps_mask(z_abs16sq, four16, _CMP_LT_OQ);
      z_im16[n] = _mm512_fmadd_ps(two16, new_im16, c_im16);
      z_re16[n] = _mm512_add_ps(new_re16, c_re16);
      // if (0 == mask)
        // break;
      // result[n] = _mm512_mask_add_epi32(result[n], mask, result[n], one16);

      // __m512 z_im16sq = _mm512_mul_ps(z_im16[n], z_im16[n]);
      // __m512 z_re16sq = _mm512_mul_ps(z_re16[n], z_re16[n]);
      // __m512 new_im16 = _mm512_mul_ps(z_re16[n], z_im16[n]);
      // __m512 z_abs16sq = _mm512_add_ps(z_re16sq, z_im16sq);
      // __m512 new_re16 = _mm512_sub_ps(z_re16sq, z_im16sq);
      // __mmask16 mask = _mm512_cmp_ps_mask(z_abs16sq, four16, _CMP_LT_OQ);
      // z_im16[n] = _mm512_fmadd_ps(two16, new_im16, c_im16);
      // z_re16[n] = _mm512_add_ps(new_re16, c_re16);
      // // if (0 == mask)
      //   // break;
      // result[n] = _mm512_mask_add_epi32(result[n], mask, result[n], one16);
    }
  }
  for (int i = 0; i < NUM; i++) {
    z_im16[0] = _mm512_add_ps(z_im16[0], z_im16[i]);
    z_re16[0] = _mm512_add_ps(z_re16[0], z_re16[i]);
  }

  int n = 0;
  __m512 z_im16sq = _mm512_mul_ps(z_im16[n], z_im16[n]);
  __m512 z_re16sq = _mm512_mul_ps(z_re16[n], z_re16[n]);
  __m512 new_im16 = _mm512_mul_ps(z_re16[n], z_im16[n]);
  __m512 z_abs16sq = _mm512_add_ps(z_re16sq, z_im16sq);
  __m512 new_re16 = _mm512_sub_ps(z_re16sq, z_im16sq);
  __mmask16 mask = _mm512_cmp_ps_mask(z_abs16sq, four16, _CMP_LT_OQ);
  result[n] = _mm512_mask_add_epi32(result[n], mask, result[n], one16);

  return result[0];
};

double avx512Mandel(float c_re, float c_im, uint32_t iter) {
  bool ret = true;
  std::vector<double> cicle_count(iter);
  
  for (size_t i = 0; i < iter; i++) {
    const int internal_iter = 100000000;
    auto start = rdtsc(); 
    auto c_re8 = _mm512_set1_ps(c_re);
    auto c_im8 = _mm512_set1_ps(c_im);

    // Run calculation of Mandelbrot fractal value for a several second (on CPU with 1GHZ)
    auto res = avx512Mandel_(c_re8, c_im8, internal_iter);
    ret &= (0 == res[0]) && (0 == res[1]) && (0 == res[2]);

    c_re += 0.01;
    c_im += 0.01;
    auto dur = rdtsc() - start;
    cicle_count[i] = double(dur) / internal_iter;
    cicle_count[i] /= 4*8; // one iteratin contains 8 independent element processing each of them has 4 multiply
  }
  std::sort(cicle_count.begin(), cicle_count.end());
  long mean_count = 0;
  for (auto count : cicle_count) mean_count += count;
  mean_count /= cicle_count.size();
  return cicle_count[cicle_count.size() / 2];
  // return mean_count;
}

#endif

#ifdef __AVX2__

/* AVX2 Implementation */
__m256i avx2Mandel_ (__m256 c_re8, __m256 c_im8, uint32_t max_iterations) {
  __m256  z_re8 = c_re8;
  __m256  z_im8 = c_im8;
  __m256  four8 = _mm256_set1_ps(4.0f);
  __m256  two8 = _mm256_set1_ps(2.0f);
  __m256i result = _mm256_set1_epi32(0);
  __m256i one8 = _mm256_set1_epi32(1);

  constexpr auto blk_size = 32;
  for (size_t b = 0; b < max_iterations / blk_size; b++)
    #pragma clang loop unroll(enable)
    #pragma GCC unroll 1
    for (auto i = 0; i < blk_size; i++) {
      __m256 z_im8sq = _mm256_mul_ps(z_im8, z_im8);
      __m256 z_re8sq = _mm256_mul_ps(z_re8, z_re8);
      __m256 new_im8 = _mm256_mul_ps(z_re8, z_im8);
      __m256 z_abs8sq = _mm256_add_ps(z_re8sq, z_im8sq);
      __m256 new_re8 = _mm256_sub_ps(z_re8sq, z_im8sq);
      __m256 mi8 = _mm256_cmp_ps(z_abs8sq, four8, _CMP_LT_OQ);
      z_im8 = _mm256_fmadd_ps(two8, new_im8, c_im8);
      z_re8 = _mm256_add_ps(new_re8, c_re8);
      int mask = _mm256_movemask_ps(mi8);
      __m256i masked1 = _mm256_and_si256(_mm256_castps_si256(mi8), one8);
      //    if (0 == mask)
      //      break;
      result = _mm256_add_epi32(result, masked1);
    }
  return result;
};

double avx2Mandel(float c_re, float c_im, uint32_t iter) {
  bool ret = true;
  std::vector<long> cicle_count(iter);

  for (size_t i = 0; i < iter; i++) {
    auto start = rdtsc(); 
    auto c_re8 = _mm256_set1_ps(c_re);
    auto c_im8 = _mm256_set1_ps(c_im);

    // Run calculation of Mandelbrot fractal value for a several second (on CPU with 1GHZ)
    auto res = avx2Mandel_(c_re8, c_im8, 1000000000);
    ret &= (0 == res[0]) && (0 == res[1]) && (0 == res[2]);

    c_re += 0.01;
    c_im += 0.01;
    auto dur = rdtsc() - start;
    // std::cout << "+ " << dur << std::endl;
  }
  return ret;
}

#else

bool avx2Mandel(float c_re, float c_im, uint32_t iter) {
  throw std::invalid_argument("AVX2 is not enabled for compiler");
  return false;
}

#endif