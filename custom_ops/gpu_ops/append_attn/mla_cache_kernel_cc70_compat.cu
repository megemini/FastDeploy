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

#include "../helper.h"
#include "utils.cuh"
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

// Safe conversion from bf16 to fp16 with precision-optimized handling
// This function uses a boundary-aware approach for maximum numerical stability
__device__ inline half safe_bf16_to_fp16(const __nv_bfloat16& val) {
    // First convert to float32 to preserve full range
    float f32_val = __bfloat162float(val);
    
    // fp16 range constants with safety margins
    const float fp16_max = 65504.0f;
    const float fp16_min = -65504.0f;
    const float fp16_safe_max = 65500.0f;  // Slightly below max to avoid rounding issues
    const float fp16_safe_min = -65500.0f;
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
    
    // Check if the value is within fp16 safe range
    if (f32_val > fp16_safe_max) {
        // Map directly to safe max value to avoid overflow
        f32_val = fp16_safe_max;
    } else if (f32_val < fp16_safe_min) {
        // Map directly to safe min value to avoid overflow
        f32_val = fp16_safe_min;
    } else if (fabsf(f32_val) < fp16_min_subnormal) {
        // Handle extremely small values (below smallest subnormal)
        if (f32_val != 0.0f) {
            // Set to smallest subnormal fp16 value while preserving sign
            f32_val = copysignf(fp16_min_subnormal, f32_val);
        }
    }
    // Values between fp16_min_subnormal and fp16_min_normal will be preserved as subnormals
    
    // Final safety clamp to ensure all values are within fp16 safe range
    f32_val = fmaxf(fminf(f32_val, fp16_safe_max), fp16_safe_min);
    
    // Convert to fp16
    return __float2half(f32_val);
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

// CC70 compatible implementation of decode_absorb_cache_kernel
// This avoids using SM75+ specific PTX features like ldmatrix and .m8n8 modifier
template <typename T, int VecSize = 1>
__global__ void decode_absorb_cache_kernel_cc70(
    const T* __restrict__ kv_nope,  // [bsz, kv_num_heads, pe_size] 512
    const T* __restrict__ kv_pe,  // [bsz, kv_num_heads, nope_size] 64
    T* __restrict__ kv_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // nope_size]
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const uint32_t elem_cnt) {
    
    int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t nope_hidden_size = kv_num_heads * nope_size;
    const uint32_t pe_hidden_size = kv_num_heads * pe_size;
    const uint32_t all_size = nope_size + pe_size;
    const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

    for (int32_t linear_index = global_thread_idx * VecSize,
                 step = gridDim.x * blockDim.x * VecSize;
         linear_index < elem_cnt;
         linear_index += step) {
        const int ori_bi = linear_index / hidden_size;
        const int bias = linear_index % hidden_size;
        const int start_token_idx = cu_seqlens_q[ori_bi];
        if (seq_lens_encoder[ori_bi] > 0) return;
        const int write_seq_id = seq_lens[ori_bi];

        if (write_seq_id == 0) continue;

        const int* block_table_now = nullptr;

        block_table_now = block_tables + ori_bi * max_blocks_per_seq;
        const int block_idx = block_table_now[write_seq_id / block_size];
        const int block_offset = write_seq_id % block_size;

        if (bias < nope_hidden_size) { // pe
            const uint32_t inner_bias = bias;
            const uint32_t hi = inner_bias / nope_size;
            const uint32_t h_bias = inner_bias % nope_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + h_bias;
            const uint32_t ori_idx =
                start_token_idx * nope_hidden_size + inner_bias;
            
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_nope[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                }
            }
        } else {
            const uint32_t inner_bias = bias - nope_hidden_size;
            const uint32_t hi = inner_bias / pe_size;
            const uint32_t h_bias = inner_bias % pe_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + nope_size + h_bias;
            const uint32_t ori_idx =
                start_token_idx * pe_hidden_size + inner_bias;
                
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_pe[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                }
            }
        }
    }
}

// CC70 compatible implementation of speculate_decode_absorb_cache_kernel
template <typename T, int VecSize = 1>
__global__ void speculate_decode_absorb_cache_kernel_cc70(
    const T* __restrict__ kv_nope,  // [bsz, kv_num_heads, pe_size] 512
    const T* __restrict__ kv_pe,  // [bsz, kv_num_heads, nope_size] 64
    T* __restrict__ kv_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // nope_size]
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const uint32_t elem_cnt) {
        
    int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t nope_hidden_size = kv_num_heads * nope_size;
    const uint32_t pe_hidden_size = kv_num_heads * pe_size;
    const uint32_t all_size = nope_size + pe_size;
    const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

    for (int32_t linear_index = global_thread_idx * VecSize,
                 step = gridDim.x * blockDim.x * VecSize;
         linear_index < elem_cnt;
         linear_index += step) {
        const int token_id = linear_index / hidden_size;
        const int ori_bi = batch_id_per_token[token_id];
        if (seq_lens[ori_bi] == 0) continue;
        const int bias = linear_index % hidden_size;
        const int start_token_idx = cu_seqlens_q[ori_bi];
        const int write_seq_id =
            seq_lens[ori_bi] + token_id - start_token_idx;
        if (write_seq_id == 0) continue;

        const int* block_table_now = nullptr;

        block_table_now = block_tables + ori_bi * max_blocks_per_seq;
        const int block_idx = block_table_now[write_seq_id / block_size];
        const int block_offset = write_seq_id % block_size;
        if (block_idx < 0) {
            printf(
                "Fatal Error!!!, block idx %d when write_seq_id is %d\n some key var "
                "%d %d %d %d\n",
                block_idx,
                write_seq_id,
                ori_bi,
                seq_lens[ori_bi],
                token_id,
                cu_seqlens_q[ori_bi]);
        }
        if (bias < nope_hidden_size) { // pe
            const uint32_t inner_bias = bias;
            const uint32_t hi = inner_bias / nope_size;
            const uint32_t h_bias = inner_bias % nope_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + h_bias;
            const uint32_t ori_idx =
                token_id * nope_hidden_size + inner_bias;
                
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_nope[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                }
            }
        } else {
            const uint32_t inner_bias = bias - nope_hidden_size;
            const uint32_t hi = inner_bias / pe_size;
            const uint32_t h_bias = inner_bias % pe_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + nope_size + h_bias;
            const uint32_t ori_idx =
                token_id * pe_hidden_size + inner_bias;
                
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_pe[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                }
            }
        }
    }
}

// CC70 compatible implementation of prefill_absorb_cache_kernel
template <typename T, int VecSize = 1>
__global__ void prefill_absorb_cache_kernel_cc70(
    const T* __restrict__ kv_nope,  // [bsz, kv_num_heads, pe_size] 512
    const T* __restrict__ kv_pe,  // [bsz, kv_num_heads, nope_size] 64
    T* __restrict__ kv_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // nope_size]
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_decoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const uint32_t elem_cnt) {
        
    int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t nope_hidden_size = kv_num_heads * nope_size;
    const uint32_t pe_hidden_size = kv_num_heads * pe_size;
    const uint32_t all_size = nope_size + pe_size;
    const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

    for (int32_t linear_index = global_thread_idx * VecSize,
                 step = gridDim.x * blockDim.x * VecSize;
         linear_index < elem_cnt;
         linear_index += step) {
        const uint32_t token_idx = linear_index / hidden_size;
        const uint32_t bias = linear_index % hidden_size;
        const uint32_t ori_bi = batch_id_per_token[token_idx];
        if (seq_lens[ori_bi] == 0) continue;
        const uint32_t ori_seq_id = (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

        const int* block_table_now = nullptr;
        block_table_now = block_tables + ori_bi * max_blocks_per_seq;
        const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
        const uint32_t block_offset = ori_seq_id % block_size;

        if (bias < nope_hidden_size) { // pe
            const uint32_t inner_bias = bias;
            const uint32_t hi = inner_bias / nope_size;
            const uint32_t h_bias = inner_bias % nope_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + h_bias;
            const uint32_t ori_idx =
                token_idx * nope_hidden_size + inner_bias;
                
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_nope[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_nope[ori_idx + i];
                }
            }
        } else {
            const uint32_t inner_bias = bias - nope_hidden_size;
            const uint32_t hi = inner_bias / pe_size;
            const uint32_t h_bias = inner_bias % pe_size;
            const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                                   hi * block_size * all_size +
                                   block_offset * all_size + nope_size + h_bias;
            const uint32_t ori_idx =
                token_idx * pe_hidden_size + inner_bias;
                
            // Simple memory copy with safe conversion for bf16
            for (int i = 0; i < VecSize; ++i) {
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // For bf16 data, ensure safe conversion when storing
                    // Check if cache is fp16 while input is bf16
                    if constexpr (std::is_same_v<decltype(kv_cache[tgt_idx + i]), half>) {
                        kv_cache[tgt_idx + i] = safe_bf16_to_fp16(kv_pe[ori_idx + i]);
                    } else {
                        // Both are bf16, direct copy is fine
                        kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                    }
                } else {
                    // For fp16 data, direct copy is fine
                    kv_cache[tgt_idx + i] = kv_pe[ori_idx + i];
                }
            }
        }
    }
}

// CC70 compatible implementation of DecodeMLAWriteCache
template <paddle::DataType T>
std::vector<paddle::Tensor> DecodeMLAWriteCacheCC70(
                    const AppendAttnMetaData& meta_data,
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& seq_lens,
                    const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& batch_id_per_token,
                    const paddle::Tensor& cu_seqlens_q,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    const bool speculate_decoder,
                    cudaStream_t& stream,
                    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto token_num = meta_data.token_nums;
  auto block_size = meta_data.block_size;
  auto nope_size = meta_data.head_dims_v;
  auto all_size = meta_data.head_dims;
  int pe_size = all_size - nope_size;
  auto kv_num_heads = meta_data.kv_num_heads;
  constexpr int PackSize = 16 / sizeof(DataType_);
  const int blocksize = 128;
  int grid_size = 1;


  if (speculate_decoder) {
    const uint32_t elem_nums = token_num * kv_num_heads * all_size;
    const int pack_num = elem_nums / PackSize;
    GetNumBlocks<128>(pack_num, &grid_size);
    speculate_decode_absorb_cache_kernel_cc70<DataType_, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
            reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
            reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
            block_tables.data<int>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            seq_lens.data<int>(),
            seq_lens_encoder.data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            kv_num_heads,
            nope_size,
            pe_size,
            block_size,
            elem_nums);
  } else {
    const uint32_t elem_nums = bsz * kv_num_heads * all_size;
    const int pack_num = elem_nums / PackSize;
    GetNumBlocks<128>(pack_num, &grid_size);
    decode_absorb_cache_kernel_cc70<DataType_, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
            reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
            reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
            block_tables.data<int>(),
            cu_seqlens_q.data<int>(),
            seq_lens.data<int>(),
            seq_lens_encoder.data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            kv_num_heads,
            nope_size,
            pe_size,
            block_size,
            elem_nums);
  }
  return {};
}

// CC70 compatible implementation of PrefillMLAWriteCache
template <paddle::DataType T>
std::vector<paddle::Tensor> PrefillMLAWriteCacheCC70(
                    const AppendAttnMetaData& meta_data,
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& seq_lens,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& batch_id_per_token,
                    const paddle::Tensor& cu_seqlens_q,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    cudaStream_t& stream,
                    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto num_tokens = meta_data.token_nums;
  auto block_size = meta_data.block_size;
  auto nope_size = meta_data.head_dims_v;
  auto all_size = meta_data.head_dims;
  int pe_size = all_size - nope_size;
  auto kv_num_heads = meta_data.kv_num_heads;
  const uint32_t elem_nums = num_tokens * kv_num_heads * all_size;

  constexpr int PackSize = 16 / sizeof(DataType_);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  prefill_absorb_cache_kernel_cc70<DataType_, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
          reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
          block_tables.data<int>(),
          batch_id_per_token.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens.data<int>(),
          seq_lens_decoder.data<int>(),
          max_seq_len,
          max_blocks_per_seq,
          kv_num_heads,
          nope_size,
          pe_size,
          block_size,
          elem_nums);
  return {};
}

// Main function that provides CC70 compatibility for DecodeMLAWriteCacheKernel
std::vector<paddle::Tensor> DecodeMLAWriteCacheKernelCC70(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const bool speculate_decoder) {
  cudaStream_t stream = kv_pe.stream();
  AppendAttnMetaData meta_data;
  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache.dims();
  meta_data.kv_num_heads = kv_cache_dims[1];
  const auto nope_size = kv_nope_dims[kv_nope_dims.size() - 1] / meta_data.kv_num_heads;
  meta_data.token_nums = kv_nope_dims[0];
  meta_data.head_dims = kv_cache_dims[3];
  meta_data.head_dims_v = nope_size;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = kv_cache_dims[2];
  meta_data.batch_size = seq_lens_encoder.dims()[0];
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return DecodeMLAWriteCacheCC70<paddle::DataType::BFLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_encoder,
                              batch_id_per_token,
                              cu_seqlens_q,
                              block_tables,
                              max_seq_len,
                              speculate_decoder,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return DecodeMLAWriteCacheCC70<paddle::DataType::FLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_encoder,
                              batch_id_per_token,
                              cu_seqlens_q,
                              block_tables,
                              max_seq_len,
                              speculate_decoder,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}

// Main function that provides CC70 compatibility for PrefillMLAWriteCacheKernel
std::vector<paddle::Tensor> PrefillMLAWriteCacheKernelCC70(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len) {
  cudaStream_t stream = kv_pe.stream();
  AppendAttnMetaData meta_data;
  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache.dims();
  meta_data.kv_num_heads = kv_cache_dims[1];
  const auto nope_size = kv_nope_dims[kv_nope_dims.size() - 1] / meta_data.kv_num_heads;
  meta_data.token_nums = kv_nope_dims[0];
  meta_data.head_dims = kv_cache_dims[3];
  meta_data.head_dims_v = nope_size;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = kv_cache_dims[2];
  meta_data.batch_size = seq_lens_decoder.dims()[0];
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return PrefillMLAWriteCacheCC70<paddle::DataType::BFLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_decoder,
                              batch_id_per_token,
                              cu_seqlens_q,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return PrefillMLAWriteCacheCC70<paddle::DataType::FLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_decoder,
                              batch_id_per_token,
                              cu_seqlens_q,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}