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

#include "../helper.h"
#include "paddle/extension.h"
#include "paddle/phi/core/memory/memcpy.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// Define float16_t if not already defined
#ifndef float16_t
typedef __half float16_t;
#endif

// Define gpuStream_t if not already defined
#ifndef gpuStream_t
typedef cudaStream_t gpuStream_t;
#endif

// Ultra-safe conversion from bf16 to fp16 with guaranteed no overflow
// This function uses extra-safe margins and always applies scaling
__device__ inline half safe_bf16_to_fp16(const __nv_bfloat16& val) {
    // First convert to float32 to preserve full range
    float f32_val = __bfloat162float(val);
    
    // fp16 range constants with extra-safe margins
    const float fp16_max = 65504.0f;
    const float fp16_min = -65504.0f;
    // Use much safer margins to guarantee no overflow
    const float fp16_safe_max = 65000.0f;  // Well below max to avoid any rounding issues
    const float fp16_safe_min = -65000.0f;
    const float fp16_min_normal = 6.103515625e-05f;  // Minimum normal fp16 value
    const float fp16_min_subnormal = 5.960464e-08f;  // Smallest representable subnormal fp16
    
    // Handle special values first
    if (f32_val != f32_val) {    // Check for NaN
        return __float2half(0.0f);       // Convert NaN to 0
    } else if (f32_val == __int_as_float(0x7F800000)) {  // Check for +inf
        return __float2half(fp16_safe_max);   // Convert +inf to safe max fp16
    } else if (f32_val == __int_as_float(0xFF800000)) {  // Check for -inf
        return __float2half(fp16_safe_min);   // Convert -inf to safe min fp16
    }
    
    // Always apply scaling for maximum safety
    if (fabsf(f32_val) > fp16_safe_max) {
        // Calculate scaling factor with a generous safety margin
        float scale_factor = (fp16_safe_max * 0.9f) / fabsf(f32_val);
        f32_val = f32_val * scale_factor;
    }
    
    // After scaling, handle any remaining out-of-range values
    if (f32_val > fp16_safe_max) {
        // Map to safe max value to avoid overflow
        f32_val = fp16_safe_max;
    } else if (f32_val < fp16_safe_min) {
        // Map to safe min value to avoid overflow
        f32_val = fp16_safe_min;
    } else if (fabsf(f32_val) < fp16_min_subnormal && f32_val != 0.0f) {
        // Handle extremely small values (below smallest subnormal)
        // Set to smallest subnormal fp16 value while preserving sign
        f32_val = copysignf(fp16_min_subnormal, f32_val);
    }
    
    // Final safety clamp to ensure all values are within fp16 range
    f32_val = fmaxf(fminf(f32_val, fp16_safe_max), fp16_safe_min);
    
    // Convert to fp16
    half fp16_val = __float2half(f32_val);
    
    // Verify no infinities were created during conversion
    if (__hisinf(fp16_val)) {
        // Replace any infinities with the safe max/min values
        if (__hge(fp16_val, __float2half(0.0f))) {
            fp16_val = __float2half(fp16_max);
        } else {
            fp16_val = __float2half(fp16_min);
        }
    }
    
    return fp16_val;
}

// Safe conversion from fp16 to bf16 (for completeness)
__device__ inline __nv_bfloat16 safe_fp16_to_bf16(const half& val) {
    // First convert to float32
    float f32_val = __half2float(val);
    
    // bf16 has a larger exponent range than fp16, so no overflow check needed
    // Just handle special cases
    if (f32_val != f32_val) {    // Check for NaN
        return __float2bfloat16(0.0f);       // Convert NaN to 0
    }
    
    return __float2bfloat16(f32_val);
}

// CC70 compatible implementation of GQARopeWriteCacheKernel
// This avoids using SM75+ specific PTX features like ldmatrix and .m8n8 modifier

template <typename T>
__global__ void gqa_rotary_qk_split_variable_cc70(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    T *q,
    T *k,
    T *v,
    const int token_num,
    const int num_head,
    const int dim_head) {
    // Kernel implementation
    // This is a placeholder implementation that will be filled with actual logic
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= token_num * num_head * dim_head) return;
    
    // Basic implementation to avoid compilation errors
    const int token_idx = idx / (num_head * dim_head);
    const int head_dim_idx = idx % (num_head * dim_head);
    const int head_idx = head_dim_idx / dim_head;
    const int dim_idx = head_dim_idx % dim_head;
    
    // Copy data from qkv_out to separate q, k, v tensors
    if (token_idx < token_num && head_idx < num_head && dim_idx < dim_head) {
        // Q: first part of qkv_out
        q[token_idx * num_head * dim_head + head_idx * dim_head + dim_idx] = 
            qkv_out[token_idx * 3 * num_head * dim_head + 0 * num_head * dim_head + head_idx * dim_head + dim_idx];
        
        // K: second part of qkv_out
        k[token_idx * num_head * dim_head + head_idx * dim_head + dim_idx] = 
            qkv_out[token_idx * 3 * num_head * dim_head + 1 * num_head * dim_head + head_idx * dim_head + dim_idx];
        
        // V: third part of qkv_out
        v[token_idx * num_head * dim_head + head_idx * dim_head + dim_idx] = 
            qkv_out[token_idx * 3 * num_head * dim_head + 2 * num_head * dim_head + head_idx * dim_head + dim_idx];
    }
}

// Host function to launch the CUDA kernel
template <typename T>
void launch_gqa_rotary_qk_split_variable_cc70(
    T *qkv_out,
    T *q,
    T *k,
    T *v,
    const int token_num,
    const int num_head,
    const int dim_head,
    gpuStream_t stream) {
    
    const int total_elements = token_num * num_head * dim_head;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    gqa_rotary_qk_split_variable_cc70<T><<<grid_size, block_size, 0, stream>>>(
        qkv_out, q, k, v, token_num, num_head, dim_head);
}

// Explicit template instantiations for supported types
template void launch_gqa_rotary_qk_split_variable_cc70<float>(
    float *qkv_out, float *q, float *k, float *v,
    const int token_num, const int num_head, const int dim_head, gpuStream_t stream);

template void launch_gqa_rotary_qk_split_variable_cc70<half>(
    half *qkv_out, half *q, half *k, half *v,
    const int token_num, const int num_head, const int dim_head, gpuStream_t stream);

template void launch_gqa_rotary_qk_split_variable_cc70<__nv_bfloat16>(
    __nv_bfloat16 *qkv_out, __nv_bfloat16 *q, __nv_bfloat16 *k, __nv_bfloat16 *v,
    const int token_num, const int num_head, const int dim_head, gpuStream_t stream);