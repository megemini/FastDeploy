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
#include <cuda_fp16.h>

// Define float16_t if not already defined
#ifndef float16_t
typedef __half float16_t;
#endif

// Define gpuStream_t if not already defined
#ifndef gpuStream_t
typedef cudaStream_t gpuStream_t;
#endif

// Forward declarations for the CC70 compatible functions
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
    const bool speculate_decoder);

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
    const int max_seq_len);

// Wrapper function that calls the CC70 compatible version
std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
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
    return DecodeMLAWriteCacheKernelCC70(
        kv_nope,
        kv_pe,
        kv_cache,
        seq_lens,
        seq_lens_encoder,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        cache_quant_type_str,
        max_seq_len,
        speculate_decoder);
}

// Wrapper function that calls the CC70 compatible version
std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
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
    return PrefillMLAWriteCacheKernelCC70(
        kv_nope,
        kv_pe,
        kv_cache,
        seq_lens,
        seq_lens_decoder,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        cache_quant_type_str,
        max_seq_len);
}

PD_BUILD_STATIC_OP(prefill_mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "seq_lens",
             "seq_lens_decoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(PrefillMLAWriteCacheKernel));

PD_BUILD_STATIC_OP(decode_mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "seq_lens",
             "seq_lens_encoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int",
            "speculate_decoder: bool"})
    .SetKernelFn(PD_KERNEL(DecodeMLAWriteCacheKernel));
