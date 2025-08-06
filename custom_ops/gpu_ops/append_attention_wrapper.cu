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

// Template definition for type2value
template <typename T>
class type2value;

template <>
class type2value<phi::dtype::bfloat16> {
    public:
    static constexpr paddle::DataType value = paddle::DataType::BFLOAT16;
};

template <>
class type2value<phi::dtype::float16> {
    public:
    static constexpr paddle::DataType value = paddle::DataType::FLOAT16;
};

// Forward declaration for the original AppendAttentionKernel
template <paddle::DataType D>
std::vector<paddle::Tensor> AppendAttentionKernel(
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
    const bool speculate_decoder);

// Forward declaration for cc70 compatibility implementation
namespace append_attention_cc70_compat {
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
        const bool speculate_decoder);
}

// Wrapper function that dispatches to the appropriate implementation based on compute capability
template <paddle::DataType D>
std::vector<paddle::Tensor> AppendAttentionKernelWrapper(
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
    
    // Get compute capability
    int device_id;
    cudaGetDevice(&device_id);
    int compute_capability;
    cudaDeviceGetAttribute(&compute_capability, cudaDevAttrComputeCapabilityMajor, device_id);
    compute_capability = compute_capability * 10;
    int minor;
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    compute_capability += minor;
    
    // Dispatch to appropriate implementation based on compute capability
    if (compute_capability >= 80) {
        // Use the original implementation for cc >= 80
        return AppendAttentionKernel<D>(
            meta_data,
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            set_max_lengths,
            max_len_kv,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_out_scales,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            out_linear_shifts,
            out_linear_smooths,
            kv_signal_data,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            quant_max_bound,
            quant_min_bound,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            speculate_decoder);
    } else if (compute_capability >= 70) {
        // Use the cc70 compatibility implementation for 70 <= cc < 80
        typedef typename PDTraits<D>::DataType DataType_;
        typedef typename PDTraits<D>::data_t data_t;
        
        if (D == paddle::DataType::FLOAT16) {
            return append_attention_cc70_compat::AppendAttentionKernelCC70<half>(
                meta_data,
                qkv,
                key_cache,
                value_cache,
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                batch_id_per_token,
                cu_seqlens_q,
                block_tables,
                encoder_batch_ids,
                encoder_tile_ids_per_batch,
                encoder_num_blocks,
                kv_batch_ids,
                kv_tile_ids_per_batch,
                kv_num_blocks,
                decoder_batch_ids,
                decoder_tile_ids_per_batch,
                decoder_num_blocks,
                set_max_lengths,
                max_len_kv,
                rotary_embs,
                attn_mask,
                qkv_bias,
                qkv_out_scales,
                cache_k_quant_scales,
                cache_v_quant_scales,
                cache_k_dequant_scales,
                cache_v_dequant_scales,
                cache_k_zp,
                cache_v_zp,
                out_linear_shifts,
                out_linear_smooths,
                kv_signal_data,
                cache_quant_type_str,
                use_neox_rotary_style,
                rope_3d,
                max_input_length,
                quant_max_bound,
                quant_min_bound,
                out_linear_in_scale,
                encoder_block_shape_q,
                decoder_block_shape_q,
                max_partition_size,
                encoder_max_partition_size,
                speculate_max_draft_token_num,
                causal,
                speculate_decoder);
        } else if (D == paddle::DataType::BFLOAT16) {
#ifdef __CUDA_BF16_TYPES_EXIST__
            return append_attention_cc70_compat::AppendAttentionKernelCC70<__nv_bfloat16>(
                meta_data,
                qkv,
                key_cache,
                value_cache,
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                batch_id_per_token,
                cu_seqlens_q,
                block_tables,
                encoder_batch_ids,
                encoder_tile_ids_per_batch,
                encoder_num_blocks,
                kv_batch_ids,
                kv_tile_ids_per_batch,
                kv_num_blocks,
                decoder_batch_ids,
                decoder_tile_ids_per_batch,
                decoder_num_blocks,
                set_max_lengths,
                max_len_kv,
                rotary_embs,
                attn_mask,
                qkv_bias,
                qkv_out_scales,
                cache_k_quant_scales,
                cache_v_quant_scales,
                cache_k_dequant_scales,
                cache_v_dequant_scales,
                cache_k_zp,
                cache_v_zp,
                out_linear_shifts,
                out_linear_smooths,
                kv_signal_data,
                cache_quant_type_str,
                use_neox_rotary_style,
                rope_3d,
                max_input_length,
                quant_max_bound,
                quant_min_bound,
                out_linear_in_scale,
                encoder_block_shape_q,
                decoder_block_shape_q,
                max_partition_size,
                encoder_max_partition_size,
                speculate_max_draft_token_num,
                causal,
                speculate_decoder);
#else
            // Fall back to FP16 if BF16 is not supported
            paddle::Tensor qkv_fp16 = GetEmptyTensor(qkv.dims(), paddle::DataType::FLOAT16, qkv.place());
            // Note: In a real implementation, we would need to convert all tensors to FP16
            // For simplicity, we'll just return an error
            PD_THROW("BFLOAT16 is not supported on this device");
            return {paddle::Tensor{}, paddle::Tensor{}};
#endif
        } else {
            PD_THROW("Unsupported data type for AppendAttention");
            return {paddle::Tensor{}, paddle::Tensor{}};
        }
    } else {
        PD_THROW("AppendAttention requires compute capability >= 70");
        return {paddle::Tensor{}, paddle::Tensor{}};
    }
}

// The main AppendAttention function that will be registered with Paddle
std::vector<paddle::Tensor> AppendAttention(
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
    const std::string& compute_dtype,
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
    
    // Create metadata
    AppendAttnMetaData meta_data;
    
    const auto& qkv_dims = qkv.dims();
    const auto& key_cache_dims = key_cache.dims();
    meta_data.token_nums = qkv_dims[0];
    meta_data.kv_num_heads = key_cache_dims[1];
    meta_data.head_dims = key_cache_dims[3];
    // TODO: trick method support c4, add attr head_dims in the future
    if (cache_quant_type_str == "cache_int4_zp") {
        meta_data.head_dims *= 2;
    }
    const int total_num_head =
        qkv_dims[qkv_dims.size() - 1] / meta_data.head_dims;
    meta_data.q_num_heads = total_num_head - 2 * meta_data.kv_num_heads;
    
    meta_data.max_blocks_per_seq = block_tables.dims()[1];
    meta_data.block_size = key_cache.dims()[2];
    meta_data.batch_size = seq_lens_this_time.dims()[0];
    
    // Dispatch by template based on compute_dtype
    auto dispatch_by_template = [&](auto temp_args) -> std::vector<paddle::Tensor> {
        return AppendAttentionKernelWrapper<type2value<decltype(temp_args)>::value>(
            meta_data,
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            set_max_lengths,
            max_len_kv,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_out_scales,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            out_linear_shifts,
            out_linear_smooths,
            kv_signal_data,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            quant_max_bound,
            quant_min_bound,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            speculate_decoder);
    };
    
    phi::dtype::float16 fp16_dtype;
    phi::dtype::bfloat16 bp16_dtype;
    
    switch (qkv.dtype()) {
        case paddle::DataType::FLOAT16: return dispatch_by_template(fp16_dtype);
        case paddle::DataType::BFLOAT16: return dispatch_by_template(bp16_dtype);
        case paddle::DataType::INT32: {
            if (compute_dtype == "bf16") {
                return dispatch_by_template(bp16_dtype);
            } else if (compute_dtype == "fp16") {
                return dispatch_by_template(fp16_dtype);
            } else {
                PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
                break;
            }
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and bfloat16 are supported. ");
            break;
        }
    }
    return {paddle::Tensor{}};
}

// Forward declaration for the original GQARopeWriteCacheKernel
std::vector<paddle::Tensor> GQARopeWriteCacheKernel(
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
    const std::string& cache_quant_type);

// Forward declaration for cc70 compatibility implementation
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
    const std::string& cache_quant_type);

// Wrapper function that dispatches to the appropriate implementation based on compute capability
std::vector<paddle::Tensor> GQARopeWriteCacheKernelWrapper(
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
    
    // Get compute capability
    int device_id;
    cudaGetDevice(&device_id);
    int compute_capability;
    cudaDeviceGetAttribute(&compute_capability, cudaDevAttrComputeCapabilityMajor, device_id);
    compute_capability = compute_capability * 10;
    int minor;
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    compute_capability += minor;
    
    // Dispatch to appropriate implementation based on compute capability
    if (compute_capability >= 80) {
        // Use the original implementation for cc >= 80
        return GQARopeWriteCacheKernel(
            qkv,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            rotary_embs,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids,
            kv_num_blocks,
            cache_batch_ids,
            cache_tile_ids,
            cache_num_blocks,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            kv_token_num,
            max_seq_len,
            cache_quant_type);
    } else if (compute_capability >= 70) {
        // Use the cc70 compatibility implementation for 70 <= cc < 80
        return GQARopeWriteCacheKernelCC70(
            qkv,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            rotary_embs,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids,
            kv_num_blocks,
            cache_batch_ids,
            cache_tile_ids,
            cache_num_blocks,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            kv_token_num,
            max_seq_len,
            cache_quant_type);
    } else {
        PD_THROW("GQARopeWriteCacheKernel requires compute capability >= 70");
        return {paddle::Tensor{}, paddle::Tensor{}, paddle::Tensor{}};
    }
}