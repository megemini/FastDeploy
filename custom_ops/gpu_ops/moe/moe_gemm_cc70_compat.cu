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

#include "moe_compatibility.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#ifdef MOE_COMPATIBILITY_CC70

namespace moe_cc70_compat {

// SIMT-based GEMM kernel for cc>=70 compatibility
template<typename T>
__global__ void moe_gemm_simt_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const T* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int* __restrict__ expert_ids,
    const int* __restrict__ token_expert_mapping,
    const int num_experts,
    const float alpha = 1.0f,
    const float beta = 0.0f) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = M * N;
    
    if (tid >= total_elements) return;
    
    const int row = tid / N;
    const int col = tid % N;
    
    // Get expert ID for this token
    const int expert_id = expert_ids[row];
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Calculate offsets for expert-specific weights
    const int expert_offset_B = expert_id * K * N;
    const int expert_offset_bias = expert_id * N;
    
    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const T a_val = A[row * K + k];
        const T b_val = B[expert_offset_B + k * N + col];
        sum += __half2float(a_val) * __half2float(b_val);
    }
    
    // Apply bias if provided
    if (bias != nullptr) {
        sum += __half2float(bias[expert_offset_bias + col]);
    }
    
    // Apply alpha and beta scaling
    if (beta != 0.0f) {
        sum = alpha * sum + beta * __half2float(C[tid]);
    } else {
        sum = alpha * sum;
    }
    
    C[tid] = __float2half(sum);
}

// Optimized SIMT kernel with shared memory for better performance
template<typename T>
__global__ void moe_gemm_simt_shared_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const T* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int* __restrict__ expert_ids,
    const int num_experts) {
    
    const int TILE_SIZE = 16;  // Optimized for cc70
    
    __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_B[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;
    
    if (row >= M || col >= N) return;
    
    const int expert_id = expert_ids[row];
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    const int expert_offset_B = expert_id * K * N;
    const int expert_offset_bias = expert_id * N;
    
    float sum = 0.0f;
    
    // Tile-based computation
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        const int k_idx = tile * TILE_SIZE + tx;
        if (k_idx < K && row < M) {
            shared_A[ty][tx] = A[row * K + k_idx];
        } else {
            shared_A[ty][tx] = __float2half(0.0f);
        }
        
        const int k_idx_b = tile * TILE_SIZE + ty;
        if (k_idx_b < K && col < N) {
            shared_B[ty][tx] = B[expert_offset_B + k_idx_b * N + col];
        } else {
            shared_B[ty][tx] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __half2float(shared_A[ty][k]) * __half2float(shared_B[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Add bias if provided
    if (bias != nullptr && col < N) {
        sum += __half2float(bias[expert_offset_bias + col]);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

// Wrapper function for MOE GEMM with cc70 compatibility
template<typename T>
cudaError_t launch_moe_gemm_cc70(
    const T* A,
    const T* B,
    T* C,
    const T* bias,
    const int M,
    const int N,
    const int K,
    const int* expert_ids,
    const int num_experts,
    cudaStream_t stream,
    bool use_shared_memory = true) {
    
    if (use_shared_memory && M >= 16 && N >= 16) {
        // Use shared memory kernel for larger matrices
        const int TILE_SIZE = 16;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        moe_gemm_simt_shared_kernel<T><<<grid, block, 0, stream>>>(
            A, B, C, bias, M, N, K, expert_ids, num_experts);
    } else {
        // Use simple kernel for smaller matrices
        const int threads_per_block = 256;
        const int total_elements = M * N;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        moe_gemm_simt_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
            A, B, C, bias, M, N, K, expert_ids, nullptr, num_experts);
    }
    
    return cudaGetLastError();
}

// Quantized GEMM kernel for INT8 weights (cc70 compatible)
template<typename T>
__global__ void moe_gemm_int8_kernel(
    const T* __restrict__ A,
    const int8_t* __restrict__ B_quant,
    const T* __restrict__ B_scales,
    T* __restrict__ C,
    const T* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int* __restrict__ expert_ids,
    const int num_experts) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = M * N;
    
    if (tid >= total_elements) return;
    
    const int row = tid / N;
    const int col = tid % N;
    
    const int expert_id = expert_ids[row];
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    const int expert_offset_B = expert_id * K * N;
    const int expert_offset_scale = expert_id * N;
    const int expert_offset_bias = expert_id * N;
    
    // Get scale for this column
    const float scale = __half2float(B_scales[expert_offset_scale + col]);
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a_val = __half2float(A[row * K + k]);
        const int8_t b_quant_val = B_quant[expert_offset_B + k * N + col];
        const float b_val = static_cast<float>(b_quant_val) * scale;
        sum += a_val * b_val;
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += __half2float(bias[expert_offset_bias + col]);
    }
    
    C[tid] = __float2half(sum);
}

// Launch quantized GEMM for cc70
template<typename T>
cudaError_t launch_moe_gemm_int8_cc70(
    const T* A,
    const int8_t* B_quant,
    const T* B_scales,
    T* C,
    const T* bias,
    const int M,
    const int N,
    const int K,
    const int* expert_ids,
    const int num_experts,
    cudaStream_t stream) {
    
    const int threads_per_block = 256;
    const int total_elements = M * N;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    moe_gemm_int8_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
        A, B_quant, B_scales, C, bias, M, N, K, expert_ids, num_experts);
    
    return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t launch_moe_gemm_cc70<__half>(
    const __half*, const __half*, __half*, const __half*,
    const int, const int, const int, const int*, const int, cudaStream_t, bool);

template cudaError_t launch_moe_gemm_int8_cc70<__half>(
    const __half*, const int8_t*, const __half*, __half*, const __half*,
    const int, const int, const int, const int*, const int, cudaStream_t);

} // namespace moe_cc70_compat

#endif // MOE_COMPATIBILITY_CC70
