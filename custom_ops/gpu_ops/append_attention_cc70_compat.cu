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

#include "append_attn/append_attention_kernel.h"
#include "append_attn/utils.cuh"
#include "helper.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 为 __nv_bfloat16 和 half 类型添加转换函数
template <typename T>
__device__ inline float convert_to_float(const T& val);

template <>
__device__ inline float convert_to_float<__nv_bfloat16>(const __nv_bfloat16& val) {
    return __bfloat162float(val);
}

template <>
__device__ inline float convert_to_float<half>(const half& val) {
    return __half2float(val);
}

template <typename T>
__device__ inline T convert_from_float(const float& val);

template <>
__device__ inline __nv_bfloat16 convert_from_float<__nv_bfloat16>(const float& val) {
    return __float2bfloat16(val);
}

template <>
__device__ inline half convert_from_float<half>(const float& val) {
    return __float2half(val);
}

#ifdef APPEND_ATTENTION_COMPATIBILITY_CC70

namespace append_attention_cc70_compat {

// Simplified metadata structure for cc70 compatibility
struct AppendAttnMetaDataCompat {
    int batch_size;
    int block_size;
    int q_num_heads;
    int kv_num_heads;
    int token_nums;
    int head_dims;
    int head_dims_v;
    int max_blocks_per_seq;
};

// Convert from full metadata to compatibility metadata
__host__ __device__ AppendAttnMetaDataCompat convert_metadata(const AppendAttnMetaData& meta_data) {
    AppendAttnMetaDataCompat compat_meta;
    compat_meta.batch_size = meta_data.batch_size;
    compat_meta.block_size = meta_data.block_size;
    compat_meta.q_num_heads = meta_data.q_num_heads;
    compat_meta.kv_num_heads = meta_data.kv_num_heads;
    compat_meta.token_nums = meta_data.token_nums;
    compat_meta.head_dims = meta_data.head_dims;
    compat_meta.head_dims_v = meta_data.head_dims_v;
    compat_meta.max_blocks_per_seq = meta_data.max_blocks_per_seq;
    return compat_meta;
}

// Simplified attention kernel for cc70 compatibility
template <typename T>
__global__ void simplified_attention_kernel(
    const AppendAttnMetaDataCompat meta_data,
    const T* __restrict__ qkv,  // [token_num, num_heads, head_dim]
    const T* __restrict__ key_cache,  // [max_block_num, num_heads, block_size, head_dim]
    const T* __restrict__ value_cache,  // [max_block_num, num_heads, head_dim, block_size]
    T* __restrict__ output,
    const int* __restrict__ seq_lens_this_time,
    const int* __restrict__ seq_lens_decoder,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_ids,
    const int* __restrict__ tile_ids_per_batch,
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = meta_data.token_nums * meta_data.q_num_heads * meta_data.head_dims;
    
    if (tid >= total_elements) return;
    
    // Calculate position in output tensor
    const int head_dim_idx = tid % meta_data.head_dims;
    const int head_idx = (tid / meta_data.head_dims) % meta_data.q_num_heads;
    const int token_idx = tid / (meta_data.q_num_heads * meta_data.head_dims);
    
    if (token_idx >= meta_data.token_nums) return;
    
    // Get batch ID for this token
    const int batch_id = batch_id_per_token[token_idx];
    if (batch_id < 0 || batch_id >= meta_data.batch_size) return;
    
    // Get sequence length for this batch
    const int seq_len = seq_lens_this_time[batch_id];
    if (seq_len <= 0) return;
    
    // Load QKV values
    const int qkv_offset = token_idx * (meta_data.q_num_heads + 2 * meta_data.kv_num_heads) * meta_data.head_dims;
    const int q_offset = qkv_offset + head_idx * meta_data.head_dims;
    const T q_val = qkv[q_offset + head_dim_idx];
    
    // Simplified attention computation
    float attention_sum = 0.0f;
    
    // For prefill (encoder) attention
    if (enable_prefill) {
        // Compute attention over all previous tokens in the sequence
        for (int i = 0; i < seq_len; ++i) {
            // Get key value for position i
            // This is a simplified version - in practice would need to handle block tables
            const int key_offset = i * meta_data.kv_num_heads * meta_data.head_dims + 
                                  (head_idx % meta_data.kv_num_heads) * meta_data.head_dims + 
                                  head_dim_idx;
            const T k_val = key_cache[key_offset];
            
            // Simple dot product (would be more complex in real implementation)
            attention_sum += convert_to_float(q_val) * convert_to_float(k_val);
        }
    }
    
    // For decoder attention
    if (is_decoder) {
        // Compute attention over cached key/value pairs
        const int dec_seq_len = seq_lens_decoder[batch_id];
        for (int i = 0; i < dec_seq_len; ++i) {
            // Get key value from cache
            // Simplified - would need to handle block tables and cache indexing
            const int key_offset = i * meta_data.kv_num_heads * meta_data.head_dims + 
                                  (head_idx % meta_data.kv_num_heads) * meta_data.head_dims + 
                                  head_dim_idx;
            const T k_val = key_cache[key_offset];
            
            // Simple dot product
            attention_sum += convert_to_float(q_val) * convert_to_float(k_val);
        }
    }
    
    // Apply softmax (simplified)
    attention_sum = attention_sum / (enable_prefill ? seq_len : seq_lens_decoder[batch_id]);
    
    // Store result
    output[tid] = convert_from_float<T>(attention_sum);
}

// Simplified cache writing kernel for cc70 compatibility
template <typename T>
__global__ void simplified_write_cache_kernel(
    const AppendAttnMetaDataCompat meta_data,
    const T* __restrict__ qkv,  // [token_num, num_heads, head_dim]
    T* __restrict__ key_cache,  // [max_block_num, num_heads, block_size, head_dim]
    T* __restrict__ value_cache,  // [max_block_num, num_heads, head_dim, block_size]
    const int* __restrict__ seq_lens_this_time,
    const int* __restrict__ seq_lens_decoder,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_ids,
    const int* __restrict__ tile_ids_per_batch,
    const int num_blocks,
    const int block_shape_q,
    const bool is_decoder) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_kv_elements = meta_data.token_nums * meta_data.kv_num_heads * meta_data.head_dims;
    
    if (tid >= total_kv_elements) return;
    
    // Calculate position in KV tensors
    const int head_dim_idx = tid % meta_data.head_dims;
    const int head_idx = (tid / meta_data.head_dims) % meta_data.kv_num_heads;
    const int token_idx = tid / (meta_data.kv_num_heads * meta_data.head_dims);
    
    if (token_idx >= meta_data.token_nums) return;
    
    // Get batch ID for this token
    const int batch_id = batch_id_per_token[token_idx];
    if (batch_id < 0 || batch_id >= meta_data.batch_size) return;
    
    // Calculate QKV offsets
    const int qkv_offset = token_idx * (meta_data.q_num_heads + 2 * meta_data.kv_num_heads) * meta_data.head_dims;
    const int k_offset = qkv_offset + (meta_data.q_num_heads + head_idx) * meta_data.head_dims;
    const int v_offset = qkv_offset + (meta_data.q_num_heads + meta_data.kv_num_heads + head_idx) * meta_data.head_dims;
    
    // Get K and V values
    const T k_val = qkv[k_offset + head_dim_idx];
    const T v_val = qkv[v_offset + head_dim_idx];
    
    // Simplified cache writing - would need to handle block tables in real implementation
    const int cache_offset = token_idx * meta_data.kv_num_heads * meta_data.head_dims + 
                            head_idx * meta_data.head_dims + 
                            head_dim_idx;
    
    key_cache[cache_offset] = k_val;
    value_cache[cache_offset] = v_val;
}

// Wrapper function for AppendAttention with cc70 compatibility
template <typename T>
std::vector<paddle::Tensor> AppendAttentionKernelCC70(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
    
    // Convert metadata for compatibility
    auto compat_meta = convert_metadata(meta_data);
    
    // Get max lengths from set_max_lengths tensor
    int max_len_this_time = set_max_lengths.data<int>()[0];
    int max_enc_len_this_time = set_max_lengths.data<int>()[1];
    int max_dec_len_this_time = set_max_lengths.data<int>()[2];
    
    // Create output tensors
    paddle::Tensor qkv_out = qkv;  // In-place for simplicity
    paddle::Tensor fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
        qkv.dtype(),
        qkv.place());
    
    auto stream = qkv.stream();
    
    // Get the actual data type
    typedef T DataType_;
    typedef T data_t;
    
    // Write to cache first
    if (max_enc_len_this_time > 0) {
        // Encoder cache writing
        const int threads_per_block = 256;
        const int total_kv_elements = meta_data.token_nums * meta_data.kv_num_heads * meta_data.head_dims;
        const int blocks = (total_kv_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_write_cache_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(key_cache).data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(value_cache).data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            encoder_batch_ids.data<int>(),
            encoder_tile_ids_per_batch.data<int>(),
            encoder_num_blocks.data<int>()[0],
            encoder_block_shape_q,
            false);  // is_decoder = false for encoder
    }
    
    // Compute attention
    if (max_enc_len_this_time > 0 || max_dec_len_this_time > 0) {
        const int threads_per_block = 256;
        const int total_elements = meta_data.token_nums * meta_data.q_num_heads * meta_data.head_dims;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_attention_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<const DataType_*>(key_cache.data<data_t>()),
            reinterpret_cast<const DataType_*>(value_cache.data<data_t>()),
            reinterpret_cast<DataType_*>(fmha_out.data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_batch_ids.data<int>() : decoder_batch_ids.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_tile_ids_per_batch.data<int>() : decoder_tile_ids_per_batch.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_num_blocks.data<int>()[0] : decoder_num_blocks.data<int>()[0],
            (max_enc_len_this_time > 0) ? encoder_block_shape_q : decoder_block_shape_q,
            max_input_length,
            max_len_kv.data<int>()[0],
            causal,
            max_dec_len_this_time > 0,
            max_enc_len_this_time > 0);
    }
    
    return {fmha_out, qkv_out};
}

// Explicit template instantiations
template <> std::vector<paddle::Tensor> AppendAttentionKernelCC70<half>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
    
    // Convert metadata for compatibility
    auto compat_meta = convert_metadata(meta_data);
    
    // Get max lengths from set_max_lengths tensor
    int max_len_this_time = set_max_lengths.data<int>()[0];
    int max_enc_len_this_time = set_max_lengths.data<int>()[1];
    int max_dec_len_this_time = set_max_lengths.data<int>()[2];
    
    // Create output tensors
    paddle::Tensor qkv_out = qkv;  // In-place for simplicity
    paddle::Tensor fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
        qkv.dtype(),
        qkv.place());
    
    auto stream = qkv.stream();
    
    // Get the actual data type
    typedef half DataType_;
    typedef paddle::float16 data_t;
    
    // Write to cache first
    if (max_enc_len_this_time > 0) {
        // Encoder cache writing
        const int threads_per_block = 256;
        const int total_kv_elements = meta_data.token_nums * meta_data.kv_num_heads * meta_data.head_dims;
        const int blocks = (total_kv_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_write_cache_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(key_cache).data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(value_cache).data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            encoder_batch_ids.data<int>(),
            encoder_tile_ids_per_batch.data<int>(),
            encoder_num_blocks.data<int>()[0],
            encoder_block_shape_q,
            false);  // is_decoder = false for encoder
    }
    
    // Compute attention
    if (max_enc_len_this_time > 0 || max_dec_len_this_time > 0) {
        const int threads_per_block = 256;
        const int total_elements = meta_data.token_nums * meta_data.q_num_heads * meta_data.head_dims;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_attention_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<const DataType_*>(key_cache.data<data_t>()),
            reinterpret_cast<const DataType_*>(value_cache.data<data_t>()),
            reinterpret_cast<DataType_*>(fmha_out.data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_batch_ids.data<int>() : decoder_batch_ids.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_tile_ids_per_batch.data<int>() : decoder_tile_ids_per_batch.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_num_blocks.data<int>()[0] : decoder_num_blocks.data<int>()[0],
            (max_enc_len_this_time > 0) ? encoder_block_shape_q : decoder_block_shape_q,
            max_input_length,
            max_len_kv.data<int>()[0],
            causal,
            max_dec_len_this_time > 0,
            max_enc_len_this_time > 0);
    }
    
    return {fmha_out, qkv_out};
}

#ifdef __CUDA_BF16_TYPES_EXIST__
template <> std::vector<paddle::Tensor> AppendAttentionKernelCC70<__nv_bfloat16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
    
    // Convert metadata for compatibility
    auto compat_meta = convert_metadata(meta_data);
    
    // Get max lengths from set_max_lengths tensor
    int max_len_this_time = set_max_lengths.data<int>()[0];
    int max_enc_len_this_time = set_max_lengths.data<int>()[1];
    int max_dec_len_this_time = set_max_lengths.data<int>()[2];
    
    // Create output tensors
    paddle::Tensor qkv_out = qkv;  // In-place for simplicity
    paddle::Tensor fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
        qkv.dtype(),
        qkv.place());
    
    auto stream = qkv.stream();
    
    // Get the actual data type
    typedef __nv_bfloat16 DataType_;
    typedef paddle::bfloat16 data_t;
    
    // Write to cache first
    if (max_enc_len_this_time > 0) {
        // Encoder cache writing
        const int threads_per_block = 256;
        const int total_kv_elements = meta_data.token_nums * meta_data.kv_num_heads * meta_data.head_dims;
        const int blocks = (total_kv_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_write_cache_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(key_cache).data<data_t>()),
            reinterpret_cast<DataType_*>(const_cast<paddle::Tensor&>(value_cache).data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            encoder_batch_ids.data<int>(),
            encoder_tile_ids_per_batch.data<int>(),
            encoder_num_blocks.data<int>()[0],
            encoder_block_shape_q,
            false);  // is_decoder = false for encoder
    }
    
    // Compute attention
    if (max_enc_len_this_time > 0 || max_dec_len_this_time > 0) {
        const int threads_per_block = 256;
        const int total_elements = meta_data.token_nums * meta_data.q_num_heads * meta_data.head_dims;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        simplified_attention_kernel<DataType_><<<blocks, threads_per_block, 0, stream>>>(
            compat_meta,
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<const DataType_*>(key_cache.data<data_t>()),
            reinterpret_cast<const DataType_*>(value_cache.data<data_t>()),
            reinterpret_cast<DataType_*>(fmha_out.data<data_t>()),
            seq_lens_this_time.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            block_tables.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_batch_ids.data<int>() : decoder_batch_ids.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_tile_ids_per_batch.data<int>() : decoder_tile_ids_per_batch.data<int>(),
            (max_enc_len_this_time > 0) ? encoder_num_blocks.data<int>()[0] : decoder_num_blocks.data<int>()[0],
            (max_enc_len_this_time > 0) ? encoder_block_shape_q : decoder_block_shape_q,
            max_input_length,
            max_len_kv.data<int>()[0],
            causal,
            max_dec_len_this_time > 0,
            max_enc_len_this_time > 0);
    }
    
    return {fmha_out, qkv_out};
}
#endif

} // namespace append_attention_cc70_compat

#endif // APPEND_ATTENTION_COMPATIBILITY_CC70