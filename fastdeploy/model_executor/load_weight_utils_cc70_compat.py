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
    Safely convert bf16 tensor to fp16 tensor using a more robust approach.
    This prevents overflow issues while preserving as much information as possible.
    
    Args:
        tensor: Input tensor (can be bf16 or other dtype)
        
    Returns:
        Converted tensor in fp16 format
    """
    if tensor is None:
        return None
        
    # Convert to numpy array for processing
    if isinstance(tensor, paddle.Tensor):
        np_array = tensor.numpy()
    else:
        np_array = np.array(tensor)
    
    # Check if the tensor is bf16
    if np_array.dtype == np.dtype('bfloat16'):
        logger.info("Converting bf16 tensor to fp16 using robust conversion")
        
        # First convert to float32 to preserve full range
        fp32_array = np_array.astype(np.float32)
        
        # Check for values that would overflow fp16
        fp16_max = 65504.0
        fp16_min = -65504.0
        
        # Count values that need special handling
        overflow_count = np.sum(np.abs(fp32_array) > fp16_max)
        if overflow_count > 0:
            logger.warning(f"Found {overflow_count} values that exceed fp16 range, applying robust conversion")
            
            # Instead of scaling, use a log-based approach to preserve relative magnitudes
            # This maintains the relative ordering of values while fitting them into fp16 range
            
            # Apply log transformation to large values to preserve relative magnitudes
            mask_large_pos = fp32_array > fp16_max
            if np.any(mask_large_pos):
                # Use log transformation for positive values
                log_vals = np.log1p(fp32_array[mask_large_pos] - fp16_max)
                # Scale log values to fit in the upper half of fp16 range
                fp32_array[mask_large_pos] = fp16_max + log_vals * (fp16_max * 0.5)
            
            mask_large_neg = fp32_array < fp16_min
            if np.any(mask_large_neg):
                # Use log transformation for negative values
                log_vals = np.log1p(-(fp32_array[mask_large_neg] - fp16_min))
                # Scale log values to fit in the lower half of fp16 range
                fp32_array[mask_large_neg] = fp16_min - log_vals * (fp16_max * 0.5)
            
            # For extreme values that still exceed range after log transformation
            mask_extreme = np.abs(fp32_array) > fp16_max * 1.5
            if np.any(mask_extreme):
                # Apply a more aggressive log transformation
                log_vals = np.log1p(np.abs(fp32_array[mask_extreme]) - fp16_max)
                scaled_vals = fp16_max + log_vals * (fp16_max * 0.25)
                fp32_array[mask_extreme] = np.sign(fp32_array[mask_extreme]) * scaled_vals
            
            # Final clamp to ensure values are within fp16 range
            fp32_array = np.clip(fp32_array, fp16_min, fp16_max)
            
            # Handle special values
            fp32_array[np.isnan(fp32_array)] = 0.0
            fp32_array[np.isposinf(fp32_array)] = fp16_max
            fp32_array[np.isneginf(fp32_array)] = fp16_min
            
            # Improved denormal handling
            # Instead of flushing to zero, scale up denormals to minimum fp16 normal
            fp16_min_normal = 6.103515625e-05  # Minimum normal fp16 value
            mask_denorm = np.abs(fp32_array) < fp16_min_normal
            if np.any(mask_denorm):
                # Scale up denormals to preserve their relative magnitudes
                denorm_vals = fp32_array[mask_denorm]
                # Find the maximum absolute value among denormals
                max_denorm = np.max(np.abs(denorm_vals))
                if max_denorm > 0:
                    # Scale all denormals proportionally
                    scale_factor = fp16_min_normal * 0.5 / max_denorm
                    fp32_array[mask_denorm] = denorm_vals * scale_factor
        
        # Convert to fp16
        fp16_array = fp32_array.astype(np.float16)
        
        # Convert back to paddle tensor
        return paddle.to_tensor(fp16_array, place=tensor.place if isinstance(tensor, paddle.Tensor) else None)
    else:
        # If not bf16, just convert to fp16 directly
        if isinstance(tensor, paddle.Tensor):
            return tensor.astype(paddle.float16)
        else:
            return paddle.to_tensor(np_array.astype(np.float16))


def deal_state_dict_cc70_compat(state_dict):
    """
    CC70 compatible version of deal_state_dict that handles bf16 to fp16 conversion.
    
    Args:
        state_dict: Model state dictionary
    """
    device = paddle.CUDAPinnedPlace()
    compute_capability = get_cuda_compute_capability()
    
    logger.info(f"Processing state_dict with CC70 compatibility (compute capability: {compute_capability})")
    
    for name, src in state_dict.items():
        if src._is_initialized() and not isinstance(src.place, paddle.CUDAPinnedPlace):
            # For CC70-79, we need to handle bf16 to fp16 conversion
            if compute_capability >= 70 and compute_capability < 80:
                if src.dtype == paddle.bfloat16:
                    logger.info(f"Converting bf16 tensor '{name}' to fp16 for CC70 compatibility")
                    # Convert bf16 to fp16 via fp32 intermediate
                    src = safe_bf16_to_fp16_tensor(src)
                    # Update the state_dict with the converted tensor
                    state_dict[name] = src
            
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
    
    # Additional CC70 compatibility processing for bf16 weights
    if compute_capability >= 70 and compute_capability < 80:
        logger.info("Applying CC70 compatibility post-processing to weights")
        compatible_dtype = get_compatible_dtype("bfloat16")
        
        if compatible_dtype == "float16":
            for name, tensor in state_dict.items():
                if isinstance(tensor, paddle.Tensor) and tensor.dtype == paddle.bfloat16:
                    logger.info(f"Post-converting bf16 tensor '{name}' to fp16")
                    state_dict[name] = safe_bf16_to_fp16_tensor(tensor)
                elif isinstance(tensor, np.ndarray) and tensor.dtype == np.dtype('bfloat16'):
                    logger.info(f"Post-converting bf16 numpy array '{name}' to fp16")
                    state_dict[name] = safe_bf16_to_fp16_tensor(tensor)
    
    return state_dict


# Import required modules that are used in the function
import os
from fastdeploy.platforms import current_platform