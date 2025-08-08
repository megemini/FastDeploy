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

// Safe conversion from bf16 to fp16 with enhanced robust overflow handling
// This function uses a sigmoid-based approach for better numerical stability
__device__ inline half safe_bf16_to_fp16(const __nv_bfloat16& val) {
    // First convert to float32 to preserve full range
    float f32_val = __bfloat162float(val);
    
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
        // This maps any positive value to a compressed range near fp16_max
        float excess = f32_val - fp16_max;
        // Normalize excess values using a more stable formula
        float normalized_excess = excess / (excess + fp16_max);  // Will be in [0, 1) range
        
        // Map to a compressed range near fp16_max
        // This ensures values very close to fp16_max stay close
        f32_val = fp16_max - fp16_min_normal * (1.0f - normalized_excess);
        
        // Final safety clamp
        f32_val = fminf(f32_val, fp16_max);
    } else if (f32_val < fp16_min) {
        // Similar approach for negative values
        float excess = fp16_min - f32_val;
        float normalized_excess = excess / (excess + fabsf(fp16_min));
        
        // Map to a compressed range near fp16_min
        f32_val = fp16_min + fp16_min_normal * (1.0f - normalized_excess);
        
        // Final safety clamp
        f32_val = fmaxf(f32_val, fp16_min);
    } else if (fabsf(f32_val) < fp16_min_normal) {
        // Improved denormal handling - preserve sign but use minimum normal value
        if (f32_val != 0.0f) {
            // Set to minimum normal fp16 value while preserving sign
            f32_val = copysignf(fp16_min_normal, f32_val);
        }
    }
    
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

// CC70 compatible implementation of GQARopeWriteCacheKernel
// This avoids using SM75+ specific PTX features like ldmatrix and .m8n8 modifier

template <typename T>
__global__ void gqa_rotary_qk_split_variable_cc70(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head) {
    
    int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_lastdim = dim_head / 2;
    const int offset = (num_heads + kv_num_heads * 2) * dim_head;
    
    if (linear_index >= token_num * offset) return;
    
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens_encoder[ori_bi] == 0) return;
    
    const int bias = linear_index % offset;
    const int hi = bias / dim_head;
    const int h_bias = bias % dim_head;
    
    const int ori_seq_id = (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];
    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;
    
    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t base_idx =
        token_idx * (num_heads + 2 * kv_num_heads) * dim_head + hi * dim_head + h_bias;
    
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < num_heads) {
        base_split_idx = token_idx * num_heads * dim_head + hi * dim_head + h_bias;
        out_p = q;
    } else if (hi < num_heads + kv_num_heads) {
        base_split_idx = kv_write_idx * kv_num_heads * dim_head + (hi - num_heads) * dim_head + h_bias;
        out_p = k;
    } else {
        out_p = v;
        base_split_idx = kv_write_idx * kv_num_heads * dim_head + (hi - num_heads - kv_num_heads) * dim_head + h_bias;
    }
    
    T val = qkv_input[base_idx];
    
    // do rope for q and k only
    if (hi < num_heads + kv_num_heads) {
        const float *cos_emb = rotary_emb;
        const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
        
        if (h_bias < half_lastdim) {
            const float cos_tmp = cos_emb[emb_idx];
            const float sin_tmp = sin_emb[emb_idx];
            
            float input_left = static_cast<float>(val);
            float input_right = static_cast<float>(qkv_input[base_idx + half_lastdim]);
            
            // Apply rotary embedding with safe conversion for bf16
            float result_left = input_left * cos_tmp - input_right * sin_tmp;
            float result_right = input_right * cos_tmp + input_left * sin_tmp;
            
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                // For bf16 data, ensure safe conversion
                // Check if the result is within bf16 range
                if (fabsf(result_left) > 3.38953139e38f) {  // bf16 max
                    val = (result_left > 0) ? static_cast<T>(3.38953139e38f) : static_cast<T>(-3.38953139e38f);
                } else {
                    val = static_cast<T>(result_left);
                }
                
                if (fabsf(result_right) > 3.38953139e38f) {  // bf16 max
                    qkv_out[base_idx + half_lastdim] = (result_right > 0) ? static_cast<T>(3.38953139e38f) : static_cast<T>(-3.38953139e38f);
                } else {
                    qkv_out[base_idx + half_lastdim] = static_cast<T>(result_right);
                }
            } else {
                // For fp16 data, direct conversion is fine
                val = static_cast<T>(result_left);
                qkv_out[base_idx + half_lastdim] = static_cast<T>(result_right);
            }
        }
    }
    
    qkv_out[base_idx] = val;
    out_p[base_split_idx] = val;
}

template <typename T>
void gqa_rotary_qk_split_variable_cc70_impl(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    gpuStream_t stream) {
    
    int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * dim_head;
    const int blocksize = 256;
    int grid_size = (elem_nums + blocksize - 1) / blocksize;
    
    gqa_rotary_qk_split_variable_cc70<T><<<grid_size, blocksize, 0, stream>>>(
        qkv_out, q, k, v, qkv_input, rotary_emb, batch_id_per_token,
        seq_lens_encoder, seq_lens_decoder, cu_seqlens_q, cu_seqlens_k,
        token_num, num_heads, kv_num_heads, seq_len, input_output_len, dim_head);
}

// Simple implementation for append_cache_kv that works on CC70
template <typename T, typename CacheT>
__global__ void append_cache_kv_cc70(
    const CacheT *__restrict__ cache_k,
    const CacheT *__restrict__ cache_v,
    T *__restrict__ k_out,
    T *__restrict__ v_out,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ cu_seqlens_k,
    const int *__restrict__ block_tables,
    const int *batch_ids,
    const int *tile_ids_per_batch,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int head_dim,
    const int block_size) {
    
    const uint32_t tile_idx = blockIdx.x, kv_head_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    
    const uint32_t batch_id = batch_ids[tile_idx];
    const uint32_t start_kv_idx = tile_ids_per_batch[tile_idx] * block_size;
    const uint32_t end_idx = seq_lens_decoder[batch_id] - start_kv_idx;
    if (seq_lens_this_time[batch_id] <= 0) {
        return;
    }
    
    const int *cur_block_table = block_tables + batch_id * max_blocks_per_seq;
    uint32_t block_id = cur_block_table[start_kv_idx / block_size];
    
    // cache_kv idx
    uint32_t kv_h_stride = block_size * head_dim;
    uint32_t block_stride = kv_num_heads * kv_h_stride;
    const CacheT *cur_cache_k = cache_k + block_id * block_stride + kv_head_idx * kv_h_stride;
    const CacheT *cur_cache_v = cache_v + block_id * block_stride + kv_head_idx * kv_h_stride;
    
    // k_out v_out idx
    uint32_t kv_t_stride = kv_num_heads * head_dim;
    T *k_write_ptr = k_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
    T *v_write_ptr = v_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
    
    // Simple memory copy without using ldmatrix
    for (uint32_t i = tid; i < block_size * head_dim; i += blockDim.x) {
        uint32_t row = i / head_dim;
        uint32_t col = i % head_dim;
        
        if (row < end_idx) {
            // Read from cache and write to output with safe conversion for bf16
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                // For bf16 data, ensure safe conversion when reading from cache
                // Check if cache is fp16 while output is bf16
                if constexpr (std::is_same_v<CacheT, half>) {
                    k_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                        safe_fp16_to_bf16(cur_cache_k[row * head_dim + col]);
                    v_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                        safe_fp16_to_bf16(cur_cache_v[row * head_dim + col]);
                } else {
                    // Both are bf16, direct copy is fine
                    k_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                        static_cast<T>(cur_cache_k[row * head_dim + col]);
                    v_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                        static_cast<T>(cur_cache_v[row * head_dim + col]);
                }
            } else {
                // For fp16 data, direct conversion is fine
                k_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                    static_cast<T>(cur_cache_k[row * head_dim + col]);
                v_write_ptr[row * kv_t_stride + kv_head_idx * head_dim + col] =
                    static_cast<T>(cur_cache_v[row * head_dim + col]);
            }
        }
    }
}

template <typename T, typename CacheT>
void append_cache_kv_cc70_impl(
    const CacheT *cache_k,
    const CacheT *cache_v,
    T *k_out,
    T *v_out,
    const int *seq_lens_this_time,
    const int *seq_lens_decoder,
    const int *cu_seqlens_k,
    const int *block_tables,
    const int *batch_ids,
    const int *tile_ids_per_batch,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int head_dim,
    const int block_size,
    const int num_tiles,
    gpuStream_t stream) {
    
    dim3 block(256);
    dim3 grid(num_tiles, kv_num_heads);
    
    append_cache_kv_cc70<T, CacheT><<<grid, block, 0, stream>>>(
        cache_k, cache_v, k_out, v_out, seq_lens_this_time, seq_lens_decoder,
        cu_seqlens_k, block_tables, batch_ids, tile_ids_per_batch,
        max_blocks_per_seq, kv_num_heads, head_dim, block_size);
}

// Main function that provides CC70 compatibility
std::vector<paddle::Tensor> GQARopeWriteCacheKernelCC70(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& cache_batch_ids,
    const paddle::Tensor& cache_tile_ids,
    const paddle::Tensor& cache_num_blocks,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const int kv_token_num,
    const int max_seq_len,
    const std::string& cache_quant_type) {
    
    auto stream = qkv.stream();
    int token_num = qkv.shape()[0];
    int seq_len = rotary_embs.shape()[3];
    
    // Extract dimensions from input tensor
    int num_heads = qkv.shape()[1] / 3;  // qkv has shape [token_num, 3, num_head, dim_head]
    int kv_num_heads = num_heads;        // Assuming same number of heads for k and v
    int head_dim = qkv.shape()[3];
    
    // Create output tensors
    auto qkv_out = paddle::empty_like(qkv);
    auto k_out = paddle::empty({token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
    auto v_out = paddle::empty({token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
    
    // Extract block information
    int max_blocks_per_seq = block_tables.shape()[1];
    int block_size = key_cache.shape()[3];  // Assuming block_size is the 4th dimension
    
    // Call CC70 compatible rotary kernel
    if (qkv.dtype() == paddle::DataType::FLOAT16) {
        // Create temporary tensors for q, k, v
        auto q = paddle::empty({token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());
        auto k = paddle::empty({token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
        auto v = paddle::empty({token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
        
        // Use const_cast to handle the const qualifier issue
        gqa_rotary_qk_split_variable_cc70_impl<half>(
            reinterpret_cast<half*>(qkv_out.data<float16_t>()),
            reinterpret_cast<half*>(q.data<float16_t>()),
            reinterpret_cast<half*>(k.data<float16_t>()),
            reinterpret_cast<half*>(v.data<float16_t>()),
            reinterpret_cast<const half*>(qkv.data<float16_t>()),  // qkv_input
            rotary_embs.data<float>(),
            batch_id_per_token.data<int>(),
            seq_lens_encoder.data<int>(),
            seq_lens_decoder.data<int>(),
            cu_seqlens_q.data<int>(),
            cu_seqlens_k.data<int>(),
            token_num,
            num_heads,
            kv_num_heads,
            seq_len,
            seq_len,
            head_dim,
            stream);
    } else {
        PD_THROW("GQARopeWriteCacheKernel only supports float16 for CC70 compatibility");
    }
    
    // Call CC70 compatible cache append kernel
    if (kv_batch_ids.data<int>() != nullptr && kv_tile_ids.data<int>() != nullptr && cache_num_blocks.data<int>() != nullptr) {
        int num_tiles = cache_num_blocks.data<int>()[0];
        
        if (cache_quant_type == "fp16") {
            append_cache_kv_cc70_impl<half, half>(
                reinterpret_cast<const half*>(key_cache.data<float16_t>()),
                reinterpret_cast<const half*>(value_cache.data<float16_t>()),
                reinterpret_cast<half*>(k_out.data<float16_t>()),
                reinterpret_cast<half*>(v_out.data<float16_t>()),
                seq_lens_this_time.data<int>(),
                seq_lens_decoder.data<int>(),
                cu_seqlens_k.data<int>(),
                block_tables.data<int>(),
                kv_batch_ids.data<int>(),
                kv_tile_ids.data<int>(),
                max_blocks_per_seq,
                kv_num_heads,
                head_dim,
                block_size,
                num_tiles,
                stream);
        } else {
            PD_THROW("GQARopeWriteCacheKernel only supports fp16 cache quant type for CC70 compatibility");
        }
    }
    
    return {qkv_out, k_out, v_out};
}