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

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef MOE_COMPATIBILITY_CC70

namespace cutlass {
namespace gemm {
namespace kernel {

// Simplified GEMM configuration for cc70 compatibility
// Avoid complex CUTLASS templates that may have version compatibility issues

// Simple type aliases for cc70 compatibility
using CC70ElementA = cutlass::half_t;
using CC70ElementB = cutlass::half_t;
using CC70ElementC = cutlass::half_t;
using CC70ElementAccumulator = float;

// Simple GEMM configuration without complex templates
struct MoeGemmCC70Config {
    static constexpr int kThreadblockM = 64;
    static constexpr int kThreadblockN = 64;
    static constexpr int kThreadblockK = 8;
    static constexpr int kWarpM = 32;
    static constexpr int kWarpN = 32;
    static constexpr int kWarpK = 8;
    static constexpr int kStages = 2;
    static constexpr int kAlignmentA = 8;
    static constexpr int kAlignmentB = 8;
};

// Simplified configurations for different data types
struct MoeGemmCC70FP16Config : public MoeGemmCC70Config {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
};

struct MoeGemmCC70INT8Config : public MoeGemmCC70Config {
    using ElementA = cutlass::half_t;
    using ElementB = int8_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = int32_t;
    static constexpr int kThreadblockK = 16;  // Larger K for INT8
    static constexpr int kWarpK = 16;
};

// Helper function to check if cc70 compatibility mode is needed
__host__ __device__ constexpr bool needs_cc70_compatibility() {
#ifdef MOE_COMPATIBILITY_CC70
    return true;
#else
    return false;
#endif
}

// Runtime compute capability check
inline bool is_cc70_device() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int cc = prop.major * 10 + prop.minor;
    return cc >= 70 && cc < 80;
}

// Simple dispatch function for cc70 compatible MOE GEMM
// Use CUBLAS instead of complex CUTLASS templates for better compatibility
template<typename T>
cudaError_t dispatch_moe_gemm_cc70(
    const T* A,
    const T* B,
    T* C,
    const T* bias,
    int M, int N, int K,
    const int* expert_ids,
    int num_experts,
    cudaStream_t stream) {

    // For cc70 compatibility, use CUBLAS-based implementation
    // This avoids complex CUTLASS template compatibility issues
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;

    // Simple GEMM call - in practice this would be batched for multiple experts
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K,
                 &beta,
                 C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);
    return cudaSuccess;
}

// Grouped GEMM dispatch for multiple experts
template<typename T>
cudaError_t dispatch_moe_grouped_gemm_cc70(
    const T** A_array,
    const T** B_array,
    T** C_array,
    const T** bias_array,
    const int* M_array,
    const int* N_array,
    const int* K_array,
    int group_count,
    cudaStream_t stream) {

    // For cc70, execute GEMMs sequentially for better compatibility
    for (int i = 0; i < group_count; ++i) {
        auto status = dispatch_moe_gemm_cc70<T>(
            A_array[i], B_array[i], C_array[i], bias_array[i],
            M_array[i], N_array[i], K_array[i],
            nullptr, 1, stream);

        if (status != cudaSuccess) {
            return status;
        }
    }

    return cudaSuccess;
}

} // namespace kernel
} // namespace gemm
} // namespace cutlass

#endif // MOE_COMPATIBILITY_CC70
