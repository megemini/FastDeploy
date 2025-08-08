// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include BF16 headers if available
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  #include <cuda_bf16.h>
  #define __CUDA_BF16_TYPES_EXIST__
#endif

// Compatibility definitions for different CUDA compute capabilities

// Check CUDA compute capability at compile time
#if defined(__CUDA_ARCH__)
  #define MOE_CUDA_ARCH __CUDA_ARCH__
#else
  #define MOE_CUDA_ARCH 700  // Default to cc70 for host code
#endif

// BF16 support detection
#if defined(MOE_COMPATIBILITY_NO_BF16) || MOE_CUDA_ARCH < 800
  #define MOE_HAS_BF16_SUPPORT 0
  // Use FP16 as fallback for BF16
  using moe_bfloat16_t = __half;
  #define MOE_BF16_TO_FP16(x) (x)
  #define MOE_FP16_TO_BF16(x) (x)
#else
  #define MOE_HAS_BF16_SUPPORT 1
  #if defined(__CUDA_BF16_TYPES_EXIST__)
    using moe_bfloat16_t = __nv_bfloat16;
    #define MOE_BF16_TO_FP16(x) __bfloat162half(x)
    #define MOE_FP16_TO_BF16(x) __half2bfloat16(x)
  #else
    // Fallback to FP16 if BF16 types don't exist
    using moe_bfloat16_t = __half;
    #define MOE_BF16_TO_FP16(x) (x)
    #define MOE_FP16_TO_BF16(x) (x)
  #endif
#endif

// Tensor Core vs SIMT selection
#if defined(MOE_COMPATIBILITY_USE_SIMT) || MOE_CUDA_ARCH < 750
  #define MOE_USE_TENSOR_CORE 0
  #define MOE_USE_SIMT 1
#else
  #define MOE_USE_TENSOR_CORE 1
  #define MOE_USE_SIMT 0
#endif

// Warp size and thread block configurations for different architectures
#if MOE_CUDA_ARCH >= 800
  #define MOE_WARP_SIZE 32
  #define MOE_MAX_THREADS_PER_BLOCK 1024
  #define MOE_PREFERRED_BLOCK_SIZE 256
#elif MOE_CUDA_ARCH >= 700
  #define MOE_WARP_SIZE 32
  #define MOE_MAX_THREADS_PER_BLOCK 1024
  #define MOE_PREFERRED_BLOCK_SIZE 128  // Smaller block size for older architectures
#else
  #define MOE_WARP_SIZE 32
  #define MOE_MAX_THREADS_PER_BLOCK 512
  #define MOE_PREFERRED_BLOCK_SIZE 128
#endif

// Memory access patterns optimization
#if MOE_CUDA_ARCH >= 800
  #define MOE_COALESCED_ACCESS_SIZE 128  // Bytes
  #define MOE_SHARED_MEMORY_SIZE 49152   // 48KB
#elif MOE_CUDA_ARCH >= 700
  #define MOE_COALESCED_ACCESS_SIZE 128  // Bytes
  #define MOE_SHARED_MEMORY_SIZE 32768   // 32KB - more conservative for cc70
#else
  #define MOE_COALESCED_ACCESS_SIZE 64   // Bytes
  #define MOE_SHARED_MEMORY_SIZE 16384   // 16KB
#endif

// Quantization support levels
#if MOE_CUDA_ARCH >= 800
  #define MOE_SUPPORTS_INT4_QUANT 1
  #define MOE_SUPPORTS_INT8_QUANT 1
  #define MOE_SUPPORTS_FP8_QUANT 0  // Requires cc>=89
#elif MOE_CUDA_ARCH >= 750
  #define MOE_SUPPORTS_INT4_QUANT 1
  #define MOE_SUPPORTS_INT8_QUANT 1
  #define MOE_SUPPORTS_FP8_QUANT 0
#elif MOE_CUDA_ARCH >= 700
  #define MOE_SUPPORTS_INT4_QUANT 0  // Limited support, use INT8 instead
  #define MOE_SUPPORTS_INT8_QUANT 1
  #define MOE_SUPPORTS_FP8_QUANT 0
#else
  #define MOE_SUPPORTS_INT4_QUANT 0
  #define MOE_SUPPORTS_INT8_QUANT 0
  #define MOE_SUPPORTS_FP8_QUANT 0
#endif

// Compatibility function declarations
namespace moe_compatibility {

// Convert BF16 to FP16 for compatibility with enhanced safety
__device__ __forceinline__ __half bf16_to_fp16_compat(const moe_bfloat16_t& bf16_val) {
#if MOE_HAS_BF16_SUPPORT
  // First convert to float32 to preserve full range
  float f32_val = __bfloat162float(bf16_val);
  
  // fp16 range constants
  const float fp16_max = 65504.0f;
  const float fp16_min = -65504.0f;
  const float fp16_min_normal = 6.103515625e-05f;  // Minimum normal fp16 value
  
  // Handle special values first
  if (f32_val != f32_val) {    // Check for NaN
      return __float2half(0.0f);       // Convert NaN to 0
  } else if (f32_val == __int_as_float(0x7F800000)) {  // Check for +inf
      return __float2half(fp16_max);   // Convert +inf to max fp16
  } else if (f32_val == __int_as_float(0xFF800000)) {  // Check for -inf
      return __float2half(fp16_min);   // Convert -inf to min fp16
  }
  
  // Check if the value is within fp16 range
  if (f32_val > fp16_max) {
      // Use sigmoid-based compression for better numerical stability
      float excess = f32_val - fp16_max;
      // Normalize excess values using a more stable formula
      float normalized_excess = excess / (excess + fp16_max);  // Will be in [0, 1) range
      
      // Map to a compressed range near fp16_max
      f32_val = fp16_max - fp16_min_normal * (1.0f - normalized_excess);
  } else if (f32_val < fp16_min) {
      // Similar approach for negative values
      float excess = fp16_min - f32_val;
      float normalized_excess = excess / (excess + fabsf(fp16_min));
      
      // Map to a compressed range near fp16_min
      f32_val = fp16_min + fp16_min_normal * (1.0f - normalized_excess);
  } else if (fabsf(f32_val) < fp16_min_normal && f32_val != 0.0f) {
      // Preserve sign but use minimum normal value
      f32_val = copysignf(fp16_min_normal, f32_val);
  }
  
  // Final safety clamp
  f32_val = fmaxf(fminf(f32_val, fp16_max), fp16_min);
  
  // Convert to fp16
  return __float2half(f32_val);
#else
  return bf16_val;  // Already FP16
#endif
}

// Convert FP16 to BF16 for compatibility
__device__ __forceinline__ moe_bfloat16_t fp16_to_bf16_compat(const __half& fp16_val) {
#if MOE_HAS_BF16_SUPPORT
  return MOE_FP16_TO_BF16(fp16_val);
#else
  return fp16_val;  // Already FP16
#endif
}

// Get optimal block size for current architecture
__host__ __device__ constexpr int get_optimal_block_size() {
  return MOE_PREFERRED_BLOCK_SIZE;
}

// Get maximum shared memory size for current architecture
__host__ __device__ constexpr int get_max_shared_memory_size() {
  return MOE_SHARED_MEMORY_SIZE;
}

// Check if Tensor Core operations are available
__host__ __device__ constexpr bool has_tensor_core_support() {
  return MOE_USE_TENSOR_CORE;
}

// Check if BF16 operations are natively supported
__host__ __device__ constexpr bool has_bf16_support() {
  return MOE_HAS_BF16_SUPPORT;
}

// Check quantization support level
__host__ __device__ constexpr bool supports_int4_quantization() {
  return MOE_SUPPORTS_INT4_QUANT;
}

__host__ __device__ constexpr bool supports_int8_quantization() {
  return MOE_SUPPORTS_INT8_QUANT;
}

__host__ __device__ constexpr bool supports_fp8_quantization() {
  return MOE_SUPPORTS_FP8_QUANT;
}

} // namespace moe_compatibility

// BF16 and FP16 arithmetic operators for cc70 compatibility
// BF16 operators - only define if BF16 is supported and not already defined
#if MOE_HAS_BF16_SUPPORT && !defined(MOE_COMPATIBILITY_BF16_OPERATORS_DEFINED)
__device__ __forceinline__ __nv_bfloat16& operator*=(__nv_bfloat16& a, float b) {
    a = __float2bfloat16(__bfloat162float(a) * b);
    return a;
}

__device__ __forceinline__ __nv_bfloat16& operator+=(__nv_bfloat16& a, const __nv_bfloat16& b) {
    a = __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
    return a;
}

__device__ __forceinline__ __nv_bfloat16& operator-=(__nv_bfloat16& a, const __nv_bfloat16& b) {
    a = __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
    return a;
}

__device__ __forceinline__ __nv_bfloat16& operator/=(__nv_bfloat16& a, const __nv_bfloat16& b) {
    a = __float2bfloat16(__bfloat162float(a) / __bfloat162float(b));
    return a;
}
#define MOE_COMPATIBILITY_BF16_OPERATORS_DEFINED
#endif // MOE_HAS_BF16_SUPPORT && !MOE_COMPATIBILITY_BF16_OPERATORS_DEFINED

// FP16 operators - only define if not already defined by CUDA toolkit
#if !defined(__CUDA_FP16_HPP__) && !defined(MOE_COMPATIBILITY_HALF_OPERATORS_DEFINED)
__device__ __forceinline__ __half& operator*=(__half& a, float b) {
    a = __float2half(__half2float(a) * b);
    return a;
}

__device__ __forceinline__ __half& operator+=(__half& a, const __half& b) {
    a = __float2half(__half2float(a) + __half2float(b));
    return a;
}

__device__ __forceinline__ __half& operator-=(__half& a, const __half& b) {
    a = __float2half(__half2float(a) - __half2float(b));
    return a;
}

__device__ __forceinline__ __half& operator/=(__half& a, const __half& b) {
    a = __float2half(__half2float(a) / __half2float(b));
    return a;
}
#define MOE_COMPATIBILITY_HALF_OPERATORS_DEFINED
#endif // !__CUDA_FP16_HPP__ && !MOE_COMPATIBILITY_HALF_OPERATORS_DEFINED

// Compatibility macros for different precision modes
#if MOE_HAS_BF16_SUPPORT
  #define MOE_PRECISION_TYPE moe_bfloat16_t
  #define MOE_PRECISION_NAME "bfloat16"
#else
  #define MOE_PRECISION_TYPE __half
  #define MOE_PRECISION_NAME "float16"
#endif

// Debug and logging macros
#ifdef MOE_COMPATIBILITY_CC70
  #define MOE_COMPATIBILITY_LOG(msg) \
    printf("[MOE Compatibility CC70] " msg "\n")
#else
  #define MOE_COMPATIBILITY_LOG(msg) \
    printf("[MOE Full Support] " msg "\n")
#endif

// Conditional compilation helpers
#define MOE_IF_CC70(code) \
  do { \
    if constexpr (MOE_CUDA_ARCH >= 700 && MOE_CUDA_ARCH < 800) { \
      code \
    } \
  } while(0)

#define MOE_IF_CC80_PLUS(code) \
  do { \
    if constexpr (MOE_CUDA_ARCH >= 800) { \
      code \
    } \
  } while(0)
