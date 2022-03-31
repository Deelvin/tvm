#include <stdio.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <stdlib.h>
#include <cstring>
using namespace std::chrono;

// void perf_check(size_t count)
// {
// #ifdef __AVX512F__
//   auto fun = [](void* pData, size_t count) {
//     float* pDataF = (float*)pData;
//     size_t num = 8;
//     __m512  a = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
//     __m512  b = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
//     __m512  c0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     __m512  c1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     __m512  c2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     __m512  c3 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     for (size_t i = 0; i < count/16/8; i++) {
//       c0 = _mm512_load_ps(pDataF);
//       c0 = _mm512_fmadd_ps(a, b, c0);
//       _mm512_store_ps(pDataF, c0);
//       pDataF += 16;

//       c1 = _mm512_load_ps(pDataF);
//       c1 = _mm512_fmadd_ps(a, b, c1);
//       _mm512_store_ps(pDataF, c1);
//       pDataF += 16;

//       c2 = _mm512_load_ps(pDataF);
//       c2 = _mm512_fmadd_ps(a, b, c2);
//       _mm512_store_ps(pDataF, c2);
//       pDataF += 16;

//       c3 = _mm512_load_ps(pDataF);
//       c3 = _mm512_fmadd_ps(a, b, c3);
//       _mm512_store_ps(pDataF, c3);
//       pDataF += 16;

//       c0 = _mm512_load_ps(pDataF);
//       c0 = _mm512_fmadd_ps(a, b, c0);
//       _mm512_store_ps(pDataF, c0);
//       pDataF += 16;

//       c1 = _mm512_load_ps(pDataF);
//       c1 = _mm512_fmadd_ps(a, b, c1);
//       _mm512_store_ps(pDataF, c1);
//       pDataF += 16;

//       c2 = _mm512_load_ps(pDataF);
//       c2 = _mm512_fmadd_ps(a, b, c2);
//       _mm512_store_ps(pDataF, c2);
//       pDataF += 16;

//       c3 = _mm512_load_ps(pDataF);
//       c3 = _mm512_fmadd_ps(a, b, c3);
//       _mm512_store_ps(pDataF, c3);
//       pDataF += 16;

//     }
//   };
// #else
// #ifdef __AVX2__
//   auto fun = [](void* pData, size_t bSize) {
//     float* pDataF = (float*)pData;
//     size_t num = 8;
//     __m256  a = {2, 1, 2, 1, 2, 1, 2, 1};
//     __m256  b = {1, 2, 1, 2, 1, 2, 1, 1};
//     __m256  c0 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c1 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c2 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c3 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c4 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c5 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c6 = {0, 0, 0, 0, 0, 0, 0, 0};
//     __m256  c7 = {0, 0, 0, 0, 0, 0, 0, 0};

//     for (size_t i = 0; i < bSize/16/8; i++) {
//       c0 = _mm256_load_ps(pDataF);
//       c1 = _mm256_load_ps(pDataF + 8);
//       c0 = _mm256_fmadd_ps(a, b, c0);
//       c1 = _mm256_fmadd_ps(a, b, c1);
//       _mm256_store_ps(pDataF, c0);
//       _mm256_store_ps(pDataF + 8, c1);
//       pDataF += 16;

//       c2 = _mm256_load_ps(pDataF);
//       c3 = _mm256_load_ps(pDataF + 8);
//       c2 = _mm256_fmadd_ps(a, b, c2);
//       c3 = _mm256_fmadd_ps(a, b, c3);
//       _mm256_store_ps(pDataF, c2);
//       _mm256_store_ps(pDataF + 8, c3);
//       pDataF += 16;

//       c4 = _mm256_load_ps(pDataF);
//       c5 = _mm256_load_ps(pDataF + 8);
//       c4 = _mm256_fmadd_ps(a, b, c4);
//       c5 = _mm256_fmadd_ps(a, b, c5);
//       _mm256_store_ps(pDataF, c4);
//       _mm256_store_ps(pDataF + 8, c5);
//       pDataF += 16;

//       c6 = _mm256_load_ps(pDataF);
//       c7 = _mm256_load_ps(pDataF + 8);
//       c6 = _mm256_fmadd_ps(a, b, c6);
//       c7 = _mm256_fmadd_ps(a, b, c7);
//       _mm256_store_ps(pDataF, c6);
//       _mm256_store_ps(pDataF + 8, c7);
//       pDataF += 16;
//       c0 = _mm256_load_ps(pDataF);
//       c1 = _mm256_load_ps(pDataF + 8);
//       c0 = _mm256_fmadd_ps(a, b, c0);
//       c1 = _mm256_fmadd_ps(a, b, c1);
//       _mm256_store_ps(pDataF, c0);
//       _mm256_store_ps(pDataF + 8, c1);
//       pDataF += 16;

//       c2 = _mm256_load_ps(pDataF);
//       c3 = _mm256_load_ps(pDataF + 8);
//       c2 = _mm256_fmadd_ps(a, b, c2);
//       c3 = _mm256_fmadd_ps(a, b, c3);
//       _mm256_store_ps(pDataF, c2);
//       _mm256_store_ps(pDataF + 8, c3);
//       pDataF += 16;

//       c4 = _mm256_load_ps(pDataF);
//       c5 = _mm256_load_ps(pDataF + 8);
//       c4 = _mm256_fmadd_ps(a, b, c4);
//       c5 = _mm256_fmadd_ps(a, b, c5);
//       _mm256_store_ps(pDataF, c4);
//       _mm256_store_ps(pDataF + 8, c5);
//       pDataF += 16;

//       c6 = _mm256_load_ps(pDataF);
//       c7 = _mm256_load_ps(pDataF + 8);
//       c6 = _mm256_fmadd_ps(a, b, c6);
//       c7 = _mm256_fmadd_ps(a, b, c7);
//       _mm256_store_ps(pDataF, c6);
//       _mm256_store_ps(pDataF + 8, c7);
//       pDataF += 16;
//     }
//   };
// #endif
// #endif
//   const size_t buffSize = 16 * 1024;
//   void* ptr = nullptr;
//   int res = posix_memalign(&ptr, 128, buffSize * sizeof(float));
//   if (res != 0)
//     return;
//   std::memset(ptr,0, buffSize * sizeof(float));
//   for (auto i = 0; i < 3; ++i) {
//     fun(ptr, buffSize);
//   }

//   high_resolution_clock::time_point start = high_resolution_clock::now();
//   std::cout << "count = " << count << "\n";
//   for (auto i = 0; i < count; ++i) {
//     fun(ptr, buffSize);
//   }
//   high_resolution_clock::time_point end = high_resolution_clock::now();
//   auto elapsed = duration_cast<nanoseconds>(end - start).count() / ((float)count);
//   std::cout << "Test duration is: " << elapsed << "\n";
//   // float* pDataF = (float*)ptr;
//   // for (size_t i = 0; i < buffSize; ++i) {
//   //   std::cout << pDataF[i] << " ";
//   // }
//   // std::cout << "\n";
//   free(ptr);
// }
void* _TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size, int dtype_code_hint,
                               int dtype_bits_hint) {
  // DLDevice dev;
  // dev.device_type = static_cast<DLDeviceType>(device_type);
  // dev.device_id = device_id;

  // DLDataType type_hint;
  // type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  // type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  // type_hint.lanes = 1;

  // return DeviceAPIManager::Get(dev)->AllocWorkspace(dev, static_cast<size_t>(size), type_hint);
  return (void*)1;
}
int _TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  return 0;
}

using _DWORD=int32_t;

#ifndef __AVX512F__
int64_t sub_AVX2(int a1, int64_t a2, int64_t *a3, __m128 _XMM0, double a5, double a6, __m128 _XMM3)
{
  int v8; // er13
  int v9; // er12
  int v10; // er13
  int64_t v12; // rbx
  unsigned int v13; // ebp
  unsigned int v16; // ecx
  int v253; // edx
  unsigned int v254; // esi
  int v353; // ecx
  bool v369; // zf
  unsigned int v371; // [rsp+Ch] [rbp-13Ch]

  v8 = (*(_DWORD *)(a2 + 8) + 6143) / *(_DWORD *)(a2 + 8);
  v9 = v8 * (a1 + 1);
  if ( v9 >= 6144 )
    v9 = 6144;
  v10 = a1 * v8;
  if ( v10 >= 6144 )
    v10 = 6144;
  // if ( v10 >= v9 )
  // {
  //   return 0;
  // }
  else
  {
    auto _R14 = *a3;
    v12 = a3[2];
    v13 = *((_DWORD *)a3 + 6);
    do
    {
      float* _RAX = (float*)_TVMBackendAllocWorkspace(1LL, v13, 4096LL, 2LL, 32LL);
      v16 = -1;
      if ( !_RAX )
        break;
      v371 = v13;
      auto  _R11 = v12;
      auto _RSI = 0LL;
      __asm__ ( "vxorps  %xmm4, %xmm4, %xmm4\n" );
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX2:::vfmadd\n");
      do
      {
        *(_DWORD *)(_RAX + 4 * _RSI) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 256) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 512) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 768) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1024) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1280) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1536) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1792) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2048) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2304) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2560) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2816) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3072) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3328) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3584) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3840) = 0;
        auto _RCX = (int)(384 * (v10 & 0xFFFFFFF8) + 384 * ((unsigned int)_RSI >> 3));
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4]\n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x20] \n"
          "vfmadd132ps ymm1, ymm4, ymmword ptr [r15+rdi*4+0x20] \n"
          "vfmadd132ps ymm0, ymm4, ymmword ptr [r15+rdi*4] \n"
        );
        auto _RBP = (4 * _RCX) | 0x40;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm2, ymmword ptr [r14+rbp] \n"
          "vmovaps ymm3, ymmword ptr [r14+rbp+0x20] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdx] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdx+0x20] \n"
        );
        auto _RDX = (4 * _RCX) | 0x80;
        __asm
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm0, ymmword ptr [r14+rdx] \n"
          "vmovaps ymm1, ymmword ptr [r14+rdx+0x20] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdx+0x20] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0xC0;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm2, ymmword ptr [r14+rdx] \n"
          "vmovaps ymm3, ymmword ptr [r14+rdx+0x20] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdx] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdx+0x20] \n"
        );
        _RDX = (4 * _RCX) | 0x100;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm0, ymmword ptr [r14+rdx] \n"
          "vmovaps ymm1, ymmword ptr [r14+rdx+0x20] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdx+0x20] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0x140;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm2, ymmword ptr [r14+rdx] \n"
          "vmovaps ymm3, ymmword ptr [r14+rdx+0x20] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdx] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdx+0x20] \n"
        );
        _RDX = (4 * _RCX) | 0x180;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm0, ymmword ptr [r14+rdx] \n"
          "vmovaps ymm1, ymmword ptr [r14+rdx+0x20] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdx+0x20] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x200] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdx] \n"
        );
        auto _RBX = (4 * _RCX) | 0x1C0;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps ymm2, ymmword ptr [r14+rbx] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+r10] \n"
          "vmovaps ymm0, ymmword ptr [r14+rbx+0x20] \n"
          "vfmadd132ps ymm0, ymm1, ymmword ptr [r15+r10+0x20] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x220] \n"
          "vfmadd132ps ymm1, ymm0, ymmword ptr [r15+rdi*4+0x220] \n"
          "vfmadd231ps ymm2, ymm3, ymmword ptr [r15+rdi*4+0x200] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x260] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x240] \n"
          "vfmadd132ps ymm3, ymm2, ymmword ptr [r15+rdi*4+0x240] \n"
          "vfmadd132ps ymm0, ymm1, ymmword ptr [r15+rdi*4+0x260] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x280] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x2A0] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x2A0] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x280] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x2E0] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x2C0] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x2C0] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x2E0] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x300] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x320] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x320] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x300] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x360] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x340] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x340] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x360] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x380] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x3A0] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x3A0] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x380] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x3E0] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x3C0] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x3C0] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x3E0] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x400] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x420] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x420] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x400] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x460] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x440] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x440] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x460] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x480] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x4A0] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x4A0] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x480] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x4E0] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x4C0] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x4C0] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x4E0] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x500] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x520] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x520] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x500] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x560] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x540] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x540] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x560] \n"
          "vmovaps ymm1, ymmword ptr [r14+rcx*4+0x580] \n"
          "vmovaps ymm2, ymmword ptr [r14+rcx*4+0x5A0] \n"
          "vfmadd132ps ymm2, ymm0, ymmword ptr [r15+rdi*4+0x5A0] \n"
          "vfmadd132ps ymm1, ymm3, ymmword ptr [r15+rdi*4+0x580] \n"
          "vmovaps ymm0, ymmword ptr [r14+rcx*4+0x5E0] \n"
          "vmovaps ymm3, ymmword ptr [r14+rcx*4+0x5C0] \n"
          "vfmadd132ps ymm3, ymm1, ymmword ptr [r15+rdi*4+0x5C0] \n"
          "vfmadd132ps ymm0, ymm2, ymmword ptr [r15+rdi*4+0x5E0] \n"
          "vmovss  dword ptr [rax+rsi*4], xmm3 \n"
          "vextractps dword ptr [rax+rsi*4+0x100], xmm3, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x200], xmm3, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x300], xmm3, 3 \n"
          "vextractf128 xmm1, ymm3, 1 \n"
          "vmovss  dword ptr [rax+rsi*4+0x400], xmm1 \n"
          "vextractps dword ptr [rax+rsi*4+0x500], xmm1, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x600], xmm1, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x700], xmm1, 3 \n"
          "vmovss  dword ptr [rax+rsi*4+0x800], xmm0 \n"
          "vextractps dword ptr [rax+rsi*4+0x900], xmm0, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x0A00], xmm0, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x0B00], xmm0, 3 \n"
          "vextractf128 xmm0, ymm0, 1 \n"
          "vmovss  dword ptr [rax+rsi*4+0x0C00], xmm0 \n"
          "vextractps dword ptr [rax+rsi*4+0x0D00], xmm0, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x0E00], xmm0, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x0F00], xmm0, 3 \n"
        );
        ++_RSI;
      }
      while ( _RSI != 64 );
      __asm__ volatile ( "# LLVM-MCA-END AVX2:::vfmadd");
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX2:::Accumulate");

      __asm__
      (
        ".intel_syntax noprefix \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x100] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x200] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x300] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x400] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x500] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x600] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x700] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x800] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x900] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F00] \n"
        "vmovups [rsp+0x148+var_138], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x120] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x220] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x320] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x420] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x520] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x620] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x720] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x820] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x920] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F20] \n"
        "vmovups [rsp+0x148+var_118], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x140] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x240] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x340] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x440] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x540] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x640] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x740] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x840] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x940] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F40] \n"
        "vmovups [rsp+0x148+var_F8], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x160] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x260] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x360] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x460] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x560] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x660] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x760] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x860] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x960] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F60] \n"
        "vmovups [rsp+0x148+var_D8], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x180] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x280] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x380] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x480] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x580] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x680] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x780] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x880] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x980] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F80] \n"
        "vmovups [rsp+0x148+var_B8], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x0A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FA0] \n"
        "vmovups [rsp+0x148+var_98], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x0C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FC0] \n"
        "vmovups [rsp+0x148+var_78], ymm0 \n"
        "vaddps  ymm0, ymm4, ymmword ptr [rax+0x0E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FE0] \n"
        "vmovups [rsp+0x148+var_58], ymm0 \n"
      );
      __asm__ volatile ( "# LLVM-MCA-END AVX2:::Accumulate");
      v253 = (8 * (int8_t)v10) & 0x38;
      v254 = (v10 << 6) & 0xFFFFFE00;
      auto _RDI = (int)(v254 + v253);
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX2:::Copy");
      __asm__ (
        ".intel_syntax noprefix \n" 
        "vmovups ymm0, [rsp+0x148+var_138] \n"
        );
      v12 = _R11;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovaps ymmword ptr [r11+rdi*4], ymm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118] \n"
      );
      _RDI = (int)(v254 + v253 + 64);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+4] \n"
      );
      _RDI = (int)(v254 + v253 + 65);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+8] \n"
      );
      _RDI = (int)(v254 + v253 + 66);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 67);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 68);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 69);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 70);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x1C] \n"
      );
      _RDI = (int)(v254 + v253 + 71);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8] \n"
      );
      _RDI = (int)(v254 + v253 + 128);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+4] \n"
      );
      _RDI = (int)(v254 + v253 + 129);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+8] \n"
      );
      _RDI = (int)(v254 + v253 + 130);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 131);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 132);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 133);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 134);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x1C] \n"
      );
      _RDI = (int)(v254 + v253 + 135);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8] \n"
      );
      _RDI = (int)(v254 + v253 + 192);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+4] \n"
      );
      _RDI = (int)(v254 + v253 + 193);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+8] \n"
      );
      _RDI = (int)(v254 + v253 + 194);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 195);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 196);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 197);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 198);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x1C] \n"
      );
      _RDI = (int)(v254 + v253 + 199);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8] \n"
      );
      _RDI = (int)(v254 + v253 + 256);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+4] \n"
      );
      _RDI = (int)(v254 + v253 + 257);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+8] \n"
      );
      _RDI = (int)(v254 + v253 + 258);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 259);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 260);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 261);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 262);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x1C] \n"
      );
      _RDI = (int)(v254 + v253 + 263);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98] \n"
      );
      _RDI = (int)(v254 + v253 + 320);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+4] \n"
      );
      _RDI = (int)(v254 + v253 + 321);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+8] \n"
      );
      _RDI = (int)(v254 + v253 + 322);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 323);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 324);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 325);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 326);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x1C] \n"
      );
      _RDI = (int)(v254 + v253 + 327);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78] \n"
      );
      _RDI = (int)(v254 + v253 + 384);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+4] \n"
      );
      _RDI = (int)(v254 + v253 + 385);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+8] \n"
      );
      _RDI = (int)(v254 + v253 + 386);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x0C] \n"
      );
      _RDI = (int)(v254 + v253 + 387);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x10] \n"
      );
      _RDI = (int)(v254 + v253 + 388);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x14] \n"
      );
      _RDI = (int)(v254 + v253 + 389);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x18] \n"
      );
      _RDI = (int)(v254 + v253 + 390);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x1C] \n"
      );
      _RSI = (int)(v254 + v253 + 391);
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rsi*4], xmm0  \n"
      );
      v353 = v253 | (v10 << 6);
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58] \n"
      );
      auto _RDX = v353 | 0x1C0;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+4] \n"
      );
      _RDX = v353 | 0x1C1;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+8] \n"
      );
      _RDX = v353 | 0x1C2;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x0C] \n"
      );
      _RDX = v353 | 0x1C3;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x10] \n"
      );
      _RDX = v353 | 0x1C4;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x14] \n"
      );
      _RDX = v353 | 0x1C5;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x18] \n"
      );
      _RDX = v353 | 0x1C6;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x1C] \n"
      );
      auto _RCX = v353 | 0x1C7;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rcx*4], xmm0 \n"
      );
      v13 = v371;
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vzeroupper \n"
      );
      __asm__ volatile ( "# LLVM-MCA-END AVX2:::Copy\n");
      v369 = (unsigned int)_TVMBackendFreeWorkspace(
                             1LL,
                             v371,
                             _RAX);
                            //  *(double *)_XMM0.m128_u64,
                            //  *(double *)&_XMM1,
                            //  a6,
                            //  *(double *)_XMM3.m128_u64,
                            //  *(double *)&_XMM4) == 0;
      v16 = -1;
      if ( !v369 )
        break;
      ++v10;
      v16 = 0;
    }
    while ( v10 != v9 );
  }
  return v16;
}
#endif

#ifdef __AVX512F__
int64_t  sub_AVX512(int a1, int64_t a2, int64_t *a3, __m128 _XMM0, double a5, double a6, __m128 _XMM3)
{
  int v8; // er13
  int v9; // er12
  int v10; // er13
  int64_t v12; // rbx
  unsigned int v13; // ebp
  unsigned int v15; // ecx
  int v205; // edx
  unsigned int v206; // esi
  int v305; // ecx
  bool v321; // zf
  unsigned int v323; // [rsp+Ch] [rbp-13Ch]

  v8 = (*(_DWORD *)(a2 + 8) + 6143) / *(_DWORD *)(a2 + 8);
  v9 = v8 * (a1 + 1);
  if ( v9 >= 6144 )
    v9 = 6144;
  v10 = a1 * v8;
  if ( v10 >= 6144 )
    v10 = 6144;
  // if ( v10 >= v9 )
  // {
  //   return 0;
  // }
  else
  {
    auto _R14 = *a3;
    v12 = a3[2];
    v13 = *((_DWORD *)a3 + 6);
    do
    {
      _DWORD* _RAX = (_DWORD*)_TVMBackendAllocWorkspace(1LL, v13, 4096LL, 2LL, 32LL);
      v15 = -1;
      if ( !_RAX )
        break;
      v323 = v13;
      auto _R11 = v12;
      auto _RSI = 0LL;
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vxorps  xmm3, xmm3, xmm3\n"
      );
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX512:::vfmadd\n");
      do
      {
        *(_DWORD *)(_RAX + 4 * _RSI) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 256) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 512) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 768) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1024) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1280) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1536) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 1792) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2048) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2304) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2560) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 2816) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3072) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3328) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3584) = 0;
        *(_DWORD *)(_RAX + 4 * _RSI + 3840) = 0;
        auto _RCX = (int)(384 * (v10 & 0xFFFFFFF8) + 384 * ((unsigned int)_RSI >> 3));
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x200] \n"
          "vfmadd132ps zmm0, zmm3, zmmword ptr [r15+rdi*4] \n"
        );
        auto _RBP = (4 * _RCX) | 0x40;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm2, zmmword ptr [r14+rbp] \n"
          "vfmadd132ps zmm2, zmm0, zmmword ptr [r15+rdx] \n"
        );
        auto _RDX = (4 * _RCX) | 0x80;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm0, zmmword ptr [r14+rdx] \n"
          "vfmadd132ps zmm0, zmm2, zmmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0xC0;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm2, zmmword ptr [r14+rdx] \n"
          "vfmadd132ps zmm2, zmm0, zmmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0x100;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm0, zmmword ptr [r14+rdx] \n"
          "vfmadd132ps zmm0, zmm2, zmmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0x140;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm2, zmmword ptr [r14+rdx] \n"
          "vfmadd132ps zmm2, zmm0, zmmword ptr [r15+rdx] \n"
        );
        _RDX = (4 * _RCX) | 0x180;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm0, zmmword ptr [r14+rdx] \n"
          "vfmadd132ps zmm0, zmm2, zmmword ptr [r15+rdx] \n"
        );
        auto _RBX = (4 * _RCX) | 0x1C0;
        __asm__
        (
          ".intel_syntax noprefix \n"
          "vmovaps zmm2, zmmword ptr [r14+rbx] \n"
          "vfmadd132ps zmm2, zmm0, zmmword ptr [r15+r10] \n"
          "vfmadd231ps zmm2, zmm1, zmmword ptr [r15+rdi*4+0x200] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x240] \n"
          "vfmadd132ps zmm0, zmm2, zmmword ptr [r15+rdi*4+0x240] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x280] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x280] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x2C0] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x2C0] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x300] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x300] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x340] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x340] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x380] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x380] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x3C0] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x3C0] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x400] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x400] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x440] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x440] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x480] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x480] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x4C0] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x4C0] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x500] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x500] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x540] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x540] \n"
          "vmovaps zmm1, zmmword ptr [r14+rcx*4+0x580] \n"
          "vfmadd132ps zmm1, zmm0, zmmword ptr [r15+rdi*4+0x580] \n"
          "vmovaps zmm0, zmmword ptr [r14+rcx*4+0x5C0] \n"
          "vfmadd132ps zmm0, zmm1, zmmword ptr [r15+rdi*4+0x5C0] \n"
          "vmovss  dword ptr [rax+rsi*4], xmm0 \n"
          "vextractps dword ptr [rax+rsi*4+0x100], xmm0, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x200], xmm0, 2 \n"
          "vextracti128 xmm1, ymm0, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x300], xmm0, 3 \n"
          "vmovd   dword ptr [rax+rsi*4+0x400], xmm1 \n"
          "vextractps dword ptr [rax+rsi*4+0x500], xmm1, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x600], xmm1, 2 \n"
          "vextracti32x4 xmm2, zmm0, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x700], xmm1, 3 \n"
          "vmovd   dword ptr [rax+rsi*4+0x800], xmm2 \n"
          "vextractps dword ptr [rax+rsi*4+0x900], xmm2, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x0A00], xmm2, 2 \n"
          "vextracti32x4 xmm0, zmm0, 3 \n"
          "vextractps dword ptr [rax+rsi*4+0x0B00], xmm2, 3 \n"
          "vmovd   dword ptr [rax+rsi*4+0x0C00], xmm0 \n"
          "vextractps dword ptr [rax+rsi*4+0x0D00], xmm0, 1 \n"
          "vextractps dword ptr [rax+rsi*4+0x0E00], xmm0, 2 \n"
          "vextractps dword ptr [rax+rsi*4+0x0F00], xmm0, 3 \n"
        );
        ++_RSI;
      }while ( _RSI != 64 );
      __asm__ volatile ( "# LLVM-MCA-END AVX512:::vfmadd\n");
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX512:::Accumulate\n");
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vpxor   xmm1, xmm1, xmm1 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x100] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x200] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x300] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x400] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x500] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x600] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x700] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x800] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x900] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E00] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F00] \n"
        "vmovups [rsp+0x148+var_138], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x120] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x220] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x320] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x420] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x520] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x620] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x720] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x820] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x920] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E20] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F20] \n"
        "vmovups [rsp+0x148+var_118], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x140] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x240] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x340] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x440] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x540] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x640] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x740] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x840] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x940] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E40] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F40] \n"
        "vmovups [rsp+0x148+var_F8], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x160] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x260] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x360] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x460] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x560] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x660] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x760] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x860] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x960] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E60] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F60] \n"
        "vmovups [rsp+0x148+var_D8], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x180] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x280] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x380] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x480] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x580] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x680] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x780] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x880] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x980] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0A80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0B80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0C80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0D80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0E80] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0F80] \n"
        "vmovups [rsp+0x148+var_B8], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x0A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9A0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EA0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FA0] \n"
        "vmovups [rsp+0x148+var_98], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x0C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9C0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EC0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FC0] \n"
        "vmovups [rsp+0x148+var_78], ymm0 \n"
        "vaddps  ymm0, ymm1, ymmword ptr [rax+0x0E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x1E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x2E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x3E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x4E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x5E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x6E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x7E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x8E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x9E0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0AE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0BE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0CE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0DE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0EE0] \n"
        "vaddps  ymm0, ymm0, ymmword ptr [rax+0x0FE0] \n"
        "vmovups [rsp+0x148+var_58], ymm0 \n"
      );
      __asm__ volatile ( "# LLVM-MCA-END AVX512:::Accumulate\n");
      v205 = (8 * (int8_t)v10) & 0x38;
      v206 = (v10 << 6) & 0xFFFFFE00;
      auto _RDI = (int)(v206 + v205);
      __asm__ volatile ( "# LLVM-MCA-BEGIN AVX512:::Copy\n");
      __asm__ 
      ( 
        ".intel_syntax noprefix \n"
        "vmovups ymm0, [rsp+0x148+var_138] \n"
      );
      v12 = _R11;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovaps ymmword ptr [r11+rdi*4], ymm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118] \n"
      );
      _RDI = (int)(v206 + v205 + 64);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+4] \n"
      );
      _RDI = (int)(v206 + v205 + 65);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+8] \n"
      );
      _RDI = (int)(v206 + v205 + 66);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 67);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 68);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 69);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 70);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_118+0x1C] \n"
      );
      _RDI = (int)(v206 + v205 + 71);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8] \n"
      );
      _RDI = (int)(v206 + v205 + 128);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+4] \n"
      );
      _RDI = (int)(v206 + v205 + 129);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+8] \n"
      );
      _RDI = (int)(v206 + v205 + 130);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 131);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 132);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 133);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 134);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_F8+0x1C] \n"
      );
      _RDI = (int)(v206 + v205 + 135);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8] \n"
      );
      _RDI = (int)(v206 + v205 + 192);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+4] \n"
      );
      _RDI = (int)(v206 + v205 + 193);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+8] \n"
      );
      _RDI = (int)(v206 + v205 + 194);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 195);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 196);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 197);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 198);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_D8+0x1C] \n"
      );
      _RDI = (int)(v206 + v205 + 199);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8] \n"
      );
      _RDI = (int)(v206 + v205 + 256);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+4] \n"
      );
      _RDI = (int)(v206 + v205 + 257);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+8] \n"
      );
      _RDI = (int)(v206 + v205 + 258);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 259);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 260);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 261);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 262);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_B8+0x1C] \n"
      );
      _RDI = (int)(v206 + v205 + 263);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98] \n"
      );
      _RDI = (int)(v206 + v205 + 320);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+4] \n"
      );
      _RDI = (int)(v206 + v205 + 321);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+8] \n"
      );
      _RDI = (int)(v206 + v205 + 322);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 323);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 324);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 325);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 326);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_98+0x1C] \n"
      );
      _RDI = (int)(v206 + v205 + 327);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78] \n"
      );
      _RDI = (int)(v206 + v205 + 384);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+4] \n"
      );
      _RDI = (int)(v206 + v205 + 385);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+8] \n"
      );
      _RDI = (int)(v206 + v205 + 386);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x0C] \n"
      );
      _RDI = (int)(v206 + v205 + 387);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x10] \n"
      );
      _RDI = (int)(v206 + v205 + 388);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x14] \n"
      );
      _RDI = (int)(v206 + v205 + 389);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x18] \n"
      );
      _RDI = (int)(v206 + v205 + 390);
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdi*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_78+0x1C] \n"
      );
      _RSI = (int)(v206 + v205 + 391);
      __asm__ (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rsi*4], xmm0 \n"
      );
      v305 = v205 | (v10 << 6);
      __asm__ (
        ".intel_syntax noprefix \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58] \n"
      );
      auto _RDX = v305 | 0x1C0;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+4] \n"
      );
      _RDX = v305 | 0x1C1;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+8] \n"
      );
      _RDX = v305 | 0x1C2;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x0C] \n"
      );
      _RDX = v305 | 0x1C3;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x10] \n"
      );
      _RDX = v305 | 0x1C4;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x14] \n"
      );
      _RDX = v305 | 0x1C5;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovss  xmm0, dword ptr [rsp+0x148+var_58+0x18] \n"
      );
      _RDX = v305 | 0x1C6;
      __asm__
      (
        ".intel_syntax noprefix \n"
        "vmovss  dword ptr [r11+rdx*4], xmm0 \n"
        "vmovd   xmm0, dword ptr [rsp+0x148+var_58+0x1C] \n"
      );
      auto _RCX = v305 | 0x1C7;
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vmovd   dword ptr [r11+rcx*4], xmm0 \n"
      );
      v13 = v323;
      __asm__ 
      (
        ".intel_syntax noprefix \n"
        "vzeroupper \n"
      );
      __asm__ volatile ( "# LLVM-MCA-END AVX512:::Copy\n");
      v321 = (unsigned int)_TVMBackendFreeWorkspace(
                             1LL,
                             v323,
                             _RAX
                            //  *(double *)_XMM0.m128_u64,
                            //  *(double *)&_XMM1,
                            //  *(double *)&_XMM2,
                            //  *(double *)_XMM3.m128_u64
                             ) == 0;
      v15 = -1;
      if ( !v321 )
        break;
      ++v10;
      v15 = 0;
    }
    while ( v10 != v9 );
  }
  return v15;
}
#endif //# __AVX512F__
int main()
{
  // size_t cnt = 1000000;
  // cnt += 1;
  // perf_check(cnt);
  const size_t buffSize = 16 * 1024;
  void* ptr = nullptr;
  int res = posix_memalign(&ptr, 128, buffSize * sizeof(float));
  if (res != 0)
    return -1;
  std::memset(ptr,0, buffSize * sizeof(float));

  // doCalc(ptr, buffSize);
  __m128 _XMM0 = {1, 2, 3, 4};
  __m128 _XMM3 = {1, 2, 3, 4};
  // avx2
  
  //avx512
#ifdef __AVX512F__
  sub_AVX512(100, 100, (int64_t*)ptr, _XMM0, 0.1f, 0.1f, _XMM3);
#else
  sub_AVX2(100, 100, (int64_t*)ptr, _XMM0, 0.1f, 0.1f, _XMM3);
#endif
  return 0;
}
