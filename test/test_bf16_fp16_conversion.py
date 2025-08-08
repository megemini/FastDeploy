#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for bf16 to fp16 conversion.
This script tests the improved safe_bf16_to_fp16_tensor function.
"""

import os
import sys
import numpy as np
import paddle

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastdeploy.model_executor.load_weight_utils_cc70_compat import safe_bf16_to_fp16_tensor
from paddleformers.utils.log import logger

def test_bf16_to_fp16_conversion():
    """Test the improved bf16 to fp16 conversion function."""
    logger.info("Testing bf16 to fp16 conversion...")
    
    # Create a range of values to test
    # Include normal values, large values, small values, and special values
    
    # Normal range values
    normal_values = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    
    # Large values beyond fp16 range
    large_values = np.array([
        1e5, 1e10, 1e20, 1e30, 
        -1e5, -1e10, -1e20, -1e30
    ], dtype=np.float32)
    
    # Small values (denormals in fp16)
    small_values = np.array([
        1e-6, 1e-7, 1e-8, 1e-10, 1e-20,
        -1e-6, -1e-7, -1e-8, -1e-10, -1e-20
    ], dtype=np.float32)
    
    # Special values
    special_values = np.array([
        0.0, np.nan, np.inf, -np.inf
    ], dtype=np.float32)
    
    # Combine all test values
    all_values = np.concatenate([normal_values, large_values, small_values, special_values])
    
    # Convert to bf16
    bf16_tensor = paddle.to_tensor(all_values).astype(paddle.bfloat16)
    
    # Convert bf16 to fp16 using our improved function
    fp16_tensor = safe_bf16_to_fp16_tensor(bf16_tensor)
    
    # Convert back to numpy for analysis
    fp16_values = fp16_tensor.numpy()
    
    # Check for NaNs or infinities in the result
    nan_count = np.sum(np.isnan(fp16_values))
    inf_count = np.sum(np.isinf(fp16_values))
    
    logger.info(f"Conversion results:")
    logger.info(f"  - Total values tested: {len(all_values)}")
    logger.info(f"  - NaN count: {nan_count}")
    logger.info(f"  - Infinity count: {inf_count}")
    
    # Check specific value ranges
    fp16_max = 65504.0
    fp16_min = -65504.0
    
    # Check if any values are outside the valid fp16 range
    out_of_range = np.sum((fp16_values > fp16_max) | (fp16_values < fp16_min))
    logger.info(f"  - Values outside fp16 range: {out_of_range}")
    
    # Check large value handling
    large_indices = np.where(np.abs(all_values) > fp16_max)[0]
    if len(large_indices) > 0:
        logger.info(f"  - Large value handling:")
        for i in large_indices[:5]:  # Show first 5 examples
            logger.info(f"    Original: {all_values[i]:.6e}, Converted: {fp16_values[i]:.6f}")
    
    # Check small value handling
    small_indices = np.where((np.abs(all_values) < 1e-5) & (all_values != 0))[0]
    if len(small_indices) > 0:
        logger.info(f"  - Small value handling:")
        for i in small_indices[:5]:  # Show first 5 examples
            logger.info(f"    Original: {all_values[i]:.6e}, Converted: {fp16_values[i]:.6e}")
    
    # Check special value handling
    special_indices = np.where(np.isnan(all_values) | np.isinf(all_values))[0]
    if len(special_indices) > 0:
        logger.info(f"  - Special value handling:")
        for i in special_indices:
            if np.isnan(all_values[i]):
                logger.info(f"    Original: NaN, Converted: {fp16_values[i]}")
            elif np.isinf(all_values[i]):
                logger.info(f"    Original: {all_values[i]}, Converted: {fp16_values[i]}")
    
    logger.info("Conversion test completed.")
    return True

if __name__ == "__main__":
    test_bf16_to_fp16_conversion()