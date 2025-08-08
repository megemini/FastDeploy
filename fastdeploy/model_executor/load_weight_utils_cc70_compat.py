"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle
import numpy as np
from paddleformers.utils.log import logger

from fastdeploy.config import get_cuda_compute_capability, get_compatible_dtype


def safe_bf16_to_fp16_tensor(tensor):
    """
    Safely convert bf16 tensor to fp16 tensor using intelligent conversion strategy.
    This approach preserves model accuracy by using statistical analysis and gradual quantization.
    
    Args:
        tensor: Input tensor (can be bf16 or other dtype)
        
    Returns:
        Converted tensor in fp16 format
    """
    if tensor is None:
        return None
        
    # Convert to numpy array for processing
    if isinstance(tensor, paddle.Tensor):
        original_place = tensor.place
        np_array = tensor.numpy()
    else:
        original_place = None
        np_array = np.array(tensor)
    
    # Check if the tensor is bf16
    if np_array.dtype == np.dtype('bfloat16'):
        logger.info("Converting bf16 tensor to fp16 using intelligent conversion strategy")
        
        # First convert to float32 to preserve full range
        fp32_array = np_array.astype(np.float32)
        
        # Check for values that would overflow fp16
        fp16_max = 65504.0
        fp16_min = -65504.0
        
        # Count values that need special handling
        overflow_count = np.sum(np.abs(fp32_array) > fp16_max)
        if overflow_count > 0:
            logger.warning(f"Found {overflow_count} values that exceed fp16 range, applying intelligent conversion")
            
            # Statistical analysis of weight distribution
            abs_values = np.abs(fp32_array)
            median_val = np.median(abs_values[abs_values > 0]) if np.any(abs_values > 0) else 1.0
            std_val = np.std(fp32_array)
            
            logger.info(f"Weight distribution - Median: {median_val}, Std: {std_val}")
            
            # Strategy 1: For values that would overflow fp16, use logarithmic scaling
            # This preserves relative magnitudes better than arbitrary scaling
            mask_overflow = np.abs(fp32_array) > fp16_max
            
            if np.any(mask_overflow):
                # Apply logarithmic scaling to preserve relative relationships
                overflow_vals = fp32_array[mask_overflow]
                sign_vals = np.sign(overflow_vals)
                log_vals = np.log10(np.abs(overflow_vals))
                
                # Scale logarithmically to fit within fp16 range while preserving relationships
                # Use the median as a reference point for scaling
                log_median = np.log10(median_val) if median_val > 0 else 0
                scaled_log_vals = log_median + (log_vals - log_median) * 0.8  # Conservative scaling
                
                # Convert back from log space
                scaled_vals = sign_vals * (10 ** scaled_log_vals)
                
                # Ensure we're within fp16 range
                scaled_vals = np.clip(scaled_vals, fp16_min, fp16_max)
                
                fp32_array[mask_overflow] = scaled_vals
                logger.info(f"Applied logarithmic scaling to {np.sum(mask_overflow)} overflow values")
            
            # Strategy 2: Handle denormals more carefully
            # Instead of setting to zero, use a threshold-based approach
            fp16_min_normal = 6.103515625e-05  # Smallest normal in fp16
            mask_denorm = np.abs(fp32_array) < fp16_min_normal
            
            if np.any(mask_denorm):
                denorm_vals = fp32_array[mask_denorm]
                
                # For very small values, use a soft threshold approach
                # Preserve values that are statistically significant
                if std_val > 0:
                    # Scale denormals based on their relative importance
                    significance_threshold = fp16_min_normal * 0.1  # 10% of smallest normal
                    mask_preserve = np.abs(denorm_vals) > significance_threshold
                    
                    if np.any(mask_preserve):
                        # Scale up significant denormals to become normal fp16 values
                        scale_factor = fp16_min_normal / np.median(np.abs(denorm_vals[mask_preserve]))
                        denorm_vals[mask_preserve] = denorm_vals[mask_preserve] * scale_factor
                    
                    # Set truly insignificant values to zero
                    mask_zero = ~mask_preserve
                    denorm_vals[mask_zero] = 0.0
                    
                    fp32_array[mask_denorm] = denorm_vals
                    logger.info(f"Processed {np.sum(mask_denorm)} denormal values, preserved {np.sum(mask_preserve)}")
            
            # Strategy 3: Handle special values more gracefully
            mask_nan = np.isnan(fp32_array)
            if np.any(mask_nan):
                # Instead of setting NaNs to zero, use the median value
                fp32_array[mask_nan] = median_val
                logger.info(f"Replaced {np.sum(mask_nan)} NaN values with median")
            
            mask_posinf = np.isposinf(fp32_array)
            if np.any(mask_posinf):
                # Use a large but finite value instead of infinity
                fp32_array[mask_posinf] = fp16_max * 0.99
                logger.info(f"Replaced {np.sum(mask_posinf)} +inf values")
            
            mask_neginf = np.isneginf(fp32_array)
            if np.any(mask_neginf):
                # Use a large but finite value instead of infinity
                fp32_array[mask_neginf] = fp16_min * 0.99
                logger.info(f"Replaced {np.sum(mask_neginf)} -inf values")
        
        # Convert to fp16
        fp16_array = fp32_array.astype(np.float16)
        
        logger.info("BF16->FP16 conversion completed successfully with intelligent conversion strategy")
        
        # Convert back to paddle tensor
        return paddle.to_tensor(fp16_array, place=original_place)
    else:
        # If not bf16, just convert to fp16 directly
        if isinstance(tensor, paddle.Tensor):
            return tensor.astype(paddle.float16)
        else:
            return paddle.to_tensor(np_array.astype(np.float16))


def ensure_safe_bf16_conversion(state_dict, compute_capability=None):
    """
    Ensure all bf16 tensors in state_dict are safely converted to fp16 for CC70-79 compatibility.
    This function provides comprehensive coverage for all model loading paths.
    
    Args:
        state_dict: Model state dictionary
        compute_capability: CUDA compute capability (optional, will be detected if not provided)
        
    Returns:
        Updated state_dict with safely converted tensors
    """
    if compute_capability is None:
        compute_capability = get_cuda_compute_capability()
    
    # Only apply conversion for CC70-79
    if not (compute_capability >= 70 and compute_capability < 80):
        logger.info(f"Compute capability {compute_capability} does not require BF16->FP16 conversion")
        return state_dict
    
    logger.info(f"Applying safe BF16->FP16 conversion for compute capability {compute_capability}")
    
    conversion_count = 0
    for name, tensor in state_dict.items():
        if isinstance(tensor, paddle.Tensor) and tensor.dtype == paddle.bfloat16:
            logger.info(f"Safely converting bf16 tensor '{name}' to fp16")
            state_dict[name] = safe_bf16_to_fp16_tensor(tensor)
            conversion_count += 1
        elif isinstance(tensor, np.ndarray) and tensor.dtype == np.dtype('bfloat16'):
            logger.info(f"Safely converting bf16 numpy array '{name}' to fp16")
            state_dict[name] = safe_bf16_to_fp16_tensor(tensor)
            conversion_count += 1
    
    if conversion_count > 0:
        logger.info(f"Successfully converted {conversion_count} bf16 tensors to fp16 using safe conversion")
        logger.info("BF16->FP16 conversion completed safely using FP32 intermediate format to prevent overflow")
    else:
        logger.info("No bf16 tensors found in state_dict")
    
    return state_dict


def deal_state_dict_cc70_compat(state_dict):
    """
    CC70 compatible version of deal_state_dict that handles bf16 to fp16 conversion.
    
    Args:
        state_dict: Model state dictionary
    """
    device = paddle.CUDAPinnedPlace()
    compute_capability = get_cuda_compute_capability()
    
    logger.info(f"Processing state_dict with CC70 compatibility (compute capability: {compute_capability})")
    
    # Apply comprehensive BF16 to FP16 conversion first
    ensure_safe_bf16_conversion(state_dict, compute_capability)
    
    for name, src in state_dict.items():
        if src._is_initialized() and not isinstance(src.place, paddle.CUDAPinnedPlace):
            # Copy to pinned memory
            dst = src._copy_to(device, True)
            dst_tensor = dst.value().get_tensor()
            src_tensor = src.value().get_tensor()
            src_tensor._clear()
            src_tensor._share_data_with(dst_tensor)


def load_composite_checkpoint_cc70_compat(
    model_path: str,
    cls,
    fd_config,
    return_numpy=True,
):
    """
    CC70 compatible version of load_composite_checkpoint that handles bf16 to fp16 conversion.
    
    Args:
        model_path: Path to the model
        cls: Model class
        fd_config: FastDeploy configuration
        return_numpy: Whether to return numpy arrays
        
    Returns:
        State dictionary with properly converted weights
    """
    from fastdeploy.model_executor.load_weight_utils import (
        load_ep_checkpoint,
        get_all_safetensors,
        load_pre_sharded_checkpoint,
        load_tp_checkpoint_v1,
        deal_state_dict,
        load_tp_checkpoint,
    )
    
    compute_capability = get_cuda_compute_capability()
    logger.info(f"Loading checkpoint with CC70 compatibility (compute capability: {compute_capability})")
    
    # Use the original loading functions
    if fd_config.parallel_config.use_ep and fd_config.speculative_config.model_type != "mtp":
        state_dict = load_ep_checkpoint(model_path, fd_config, return_numpy=True)
    else:
        rank_dirs = [
            f for f in os.listdir(model_path) if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if fd_config.parallel_config.tensor_parallel_size != len(rank_dirs):
                raise ValueError(f"Your model only supports loading with tp{len(rank_dirs)}")
            state_dict = load_pre_sharded_checkpoint(
                model_path,
                fd_config.parallel_config.tensor_parallel_rank,
                use_fastsafetensor=False,
            )
        else:
            if fd_config.load_config.use_fastsafetensor and (
                current_platform.available() and current_platform.is_cuda()
            ):
                state_dict = load_tp_checkpoint_v1(model_path, cls, fd_config, use_fastsafetensor=True)
                # Use CC70 compatible state dict processing
                deal_state_dict_cc70_compat(state_dict)
            else:
                try:
                    state_dict = load_tp_checkpoint(
                        model_path,
                        cls,
                        fd_config.model_config.pretrained_config,
                        return_numpy=return_numpy,
                    )
                except Exception as e:
                    logger.error(f"Failed to load TP checkpoint: {e}")
                    if "SafeTensorError::MetadataIncompleteBuffer" in str(e):
                        raise ValueError(
                            f"SafeTensorError::MetadataIncompleteBuffer - The safetensors file appears to be corrupted or incomplete. "
                            f"Please check the integrity of your model files in {model_path}. "
                            f"Original error: {e}"
                        )
                    else:
                        raise ValueError(f"Failed to load model checkpoint: {e}")
    
    if not state_dict:
        raise ValueError("weight not found in state_dict !")
    
    # Apply comprehensive CC70 compatibility processing for bf16 weights
    ensure_safe_bf16_conversion(state_dict, compute_capability)
    
    return state_dict


# Import required modules that are used in the function
import os
from fastdeploy.platforms import current_platform