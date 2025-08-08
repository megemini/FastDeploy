#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced test script for bf16 to fp16 conversion.
This script tests the improved safe_bf16_to_fp16_tensor function with comprehensive test cases.
"""

import os
import sys
import numpy as np
import paddle
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastdeploy.model_executor.load_weight_utils_cc70_compat import safe_bf16_to_fp16_tensor
from paddleformers.utils.log import logger

def test_bf16_to_fp16_conversion():
    """Test the enhanced bf16 to fp16 conversion function."""
    logger.info("Testing enhanced bf16 to fp16 conversion...")
    
    # Create a comprehensive range of values to test
    
    # Normal range values with fine granularity
    normal_values = np.linspace(-100.0, 100.0, 200).astype(np.float32)
    
    # Values near fp16 limits with precise boundary testing
    near_limit_values = np.array([
        65490.0, 65495.0, 65499.0, 65500.0, 65501.0, 65502.0, 65503.0, 65504.0, 65505.0, 65510.0,
        -65490.0, -65495.0, -65499.0, -65500.0, -65501.0, -65502.0, -65503.0, -65504.0, -65505.0, -65510.0
    ], dtype=np.float32)
    
    # Large values beyond fp16 range (with more gradual scaling)
    large_values = np.array([
        1e4, 5e4, 1e5, 5e5, 1e6, 1e7, 1e10, 1e20, 1e30, 
        -1e4, -5e4, -1e5, -5e5, -1e6, -1e7, -1e10, -1e20, -1e30
    ], dtype=np.float32)
    
    # Small values (denormals in fp16) with more test cases
    small_values = np.array([
        1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 1e-10, 1e-15, 1e-20, 1e-30,
        -1e-5, -5e-6, -1e-6, -5e-7, -1e-7, -5e-8, -1e-8, -1e-10, -1e-15, -1e-20, -1e-30
    ], dtype=np.float32)
    
    # Special values
    special_values = np.array([
        0.0, np.nan, np.inf, -np.inf
    ], dtype=np.float32)
    
    # Combine all test values
    all_values = np.concatenate([normal_values, near_limit_values, large_values, small_values, special_values])
    
    # Convert to bf16
    bf16_tensor = paddle.to_tensor(all_values).astype(paddle.bfloat16)
    
    # Measure conversion time
    start_time = time.time()
    
    # Convert bf16 to fp16 using our enhanced function
    fp16_tensor = safe_bf16_to_fp16_tensor(bf16_tensor)
    
    conversion_time = time.time() - start_time
    
    # Convert back to numpy for analysis
    fp16_values = fp16_tensor.numpy()
    
    # Check for NaNs or infinities in the result
    nan_count = np.sum(np.isnan(fp16_values))
    inf_count = np.sum(np.isinf(fp16_values))
    
    logger.info(f"Enhanced conversion results:")
    logger.info(f"  - Total values tested: {len(all_values)}")
    logger.info(f"  - Conversion time: {conversion_time:.6f} seconds")
    logger.info(f"  - NaN count: {nan_count}")
    logger.info(f"  - Infinity count: {inf_count}")
    
    # Check specific value ranges
    fp16_max = 65504.0
    fp16_min = -65504.0
    fp16_min_normal = 6.103515625e-05  # Minimum normal fp16 value
    
    # Check if any values are outside the valid fp16 range
    out_of_range = np.sum((fp16_values > fp16_max) | (fp16_values < fp16_min))
    logger.info(f"  - Values outside fp16 range: {out_of_range}")
    
    # Check zero preservation
    zero_indices = np.where(all_values == 0.0)[0]
    zero_preserved = np.sum(fp16_values[zero_indices] == 0.0)
    logger.info(f"  - Zero preservation: {zero_preserved}/{len(zero_indices)}")
    
    # Check sign preservation for all non-zero values
    non_zero_indices = np.where((all_values != 0.0) & np.isfinite(all_values))[0]
    sign_preserved = np.sum(np.sign(fp16_values[non_zero_indices]) == np.sign(all_values[non_zero_indices]))
    logger.info(f"  - Sign preservation: {sign_preserved}/{len(non_zero_indices)}")
    
    # Check large value handling
    large_indices = np.where(np.abs(all_values) > fp16_max)[0]
    if len(large_indices) > 0:
        logger.info(f"  - Large value handling ({len(large_indices)} values):")
        for i in large_indices[:10]:  # Show first 10 examples
            logger.info(f"    Original: {all_values[i]:.6e}, Converted: {fp16_values[i]:.6f}")
    
    # Check near-limit value handling
    near_limit_indices = np.where((np.abs(all_values) > fp16_max * 0.99) & (np.abs(all_values) <= fp16_max))[0]
    if len(near_limit_indices) > 0:
        logger.info(f"  - Near-limit value handling ({len(near_limit_indices)} values):")
        for i in near_limit_indices[:5]:  # Show first 5 examples
            logger.info(f"    Original: {all_values[i]:.6f}, Converted: {fp16_values[i]:.6f}")
    
    # Check small value handling
    small_indices = np.where((np.abs(all_values) < fp16_min_normal) & (all_values != 0.0))[0]
    if len(small_indices) > 0:
        logger.info(f"  - Small value handling ({len(small_indices)} values):")
        for i in small_indices[:10]:  # Show first 10 examples
            logger.info(f"    Original: {all_values[i]:.6e}, Converted: {fp16_values[i]:.6e}")
    
    # Check special value handling
    special_indices = np.where(np.isnan(all_values) | np.isinf(all_values))[0]
    if len(special_indices) > 0:
        logger.info(f"  - Special value handling ({len(special_indices)} values):")
        for i in special_indices:
            if np.isnan(all_values[i]):
                logger.info(f"    Original: NaN, Converted: {fp16_values[i]}")
            elif np.isinf(all_values[i]):
                logger.info(f"    Original: {all_values[i]}, Converted: {fp16_values[i]}")
    
    # Check relative ordering preservation for normal values
    normal_indices = np.where((np.abs(all_values) <= fp16_max) & (np.abs(all_values) >= fp16_min_normal))[0]
    if len(normal_indices) >= 2:
        original_diffs = np.diff(all_values[normal_indices])
        converted_diffs = np.diff(fp16_values[normal_indices])
        ordering_preserved = np.sum((original_diffs > 0) == (converted_diffs > 0))
        logger.info(f"  - Relative ordering preservation: {ordering_preserved}/{len(original_diffs)} ({ordering_preserved/len(original_diffs)*100:.2f}%)")
    
    logger.info("Enhanced conversion test completed.")
    return True

def test_real_world_scenario():
    """Test with a more realistic tensor that might be found in a model."""
    logger.info("\nTesting with realistic model tensor...")
    
    # Create a tensor with a distribution similar to model weights
    # Most values small, some medium, few large outliers
    
    # Generate a normal distribution for most values
    np.random.seed(42)  # For reproducibility
    n_elements = 10000
    
    # Create a tensor with mixed value ranges
    tensor_values = np.random.normal(0, 0.1, n_elements).astype(np.float32)
    
    # Add some medium values
    medium_indices = np.random.choice(n_elements, size=int(n_elements * 0.05), replace=False)
    tensor_values[medium_indices] = np.random.uniform(1.0, 100.0, size=len(medium_indices))
    
    # Add a few large outliers
    large_indices = np.random.choice(n_elements, size=int(n_elements * 0.01), replace=False)
    tensor_values[large_indices] = np.random.uniform(1e4, 1e8, size=len(large_indices))
    
    # Add some very small values
    small_indices = np.random.choice(n_elements, size=int(n_elements * 0.1), replace=False)
    tensor_values[small_indices] = np.random.uniform(1e-10, 1e-5, size=len(small_indices))
    
    # Convert to bf16
    bf16_tensor = paddle.to_tensor(tensor_values).astype(paddle.bfloat16)
    
    # Convert bf16 to fp16 using our enhanced function
    fp16_tensor = safe_bf16_to_fp16_tensor(bf16_tensor)
    
    # Convert back to numpy for analysis
    fp16_values = fp16_tensor.numpy()
    
    # Basic statistics
    logger.info(f"  - Original tensor stats: min={np.min(tensor_values):.6e}, max={np.max(tensor_values):.6e}")
    logger.info(f"  - Converted tensor stats: min={np.min(fp16_values):.6e}, max={np.max(fp16_values):.6e}")
    
    # Check for NaNs or infinities in the result
    nan_count = np.sum(np.isnan(fp16_values))
    inf_count = np.sum(np.isinf(fp16_values))
    logger.info(f"  - NaN count: {nan_count}")
    logger.info(f"  - Infinity count: {inf_count}")
    
    # Check value preservation
    fp16_max = 65504.0
    fp16_min = -65504.0
    fp16_min_normal = 6.103515625e-05
    
    # Check normal range values
    normal_mask = (np.abs(tensor_values) <= fp16_max) & (np.abs(tensor_values) >= fp16_min_normal)
    normal_count = np.sum(normal_mask)
    normal_error = np.mean(np.abs(fp16_values[normal_mask] - tensor_values[normal_mask]) / (np.abs(tensor_values[normal_mask]) + 1e-10))
    logger.info(f"  - Normal range values ({normal_count}): mean relative error = {normal_error:.6f}")
    
    # Check large values
    large_mask = np.abs(tensor_values) > fp16_max
    large_count = np.sum(large_mask)
    logger.info(f"  - Large values ({large_count}): all within fp16 range = {np.all(np.abs(fp16_values[large_mask]) <= fp16_max)}")
    
    # Check small values
    small_mask = (np.abs(tensor_values) < fp16_min_normal) & (tensor_values != 0)
    small_count = np.sum(small_mask)
    small_preserved = np.sum(fp16_values[small_mask] != 0)
    logger.info(f"  - Small values ({small_count}): preserved non-zero = {small_preserved} ({small_preserved/small_count*100:.2f}%)")
    
    # Check sign preservation
    non_zero_mask = tensor_values != 0
    sign_preserved = np.sum(np.sign(fp16_values[non_zero_mask]) == np.sign(tensor_values[non_zero_mask]))
    logger.info(f"  - Sign preservation: {sign_preserved}/{np.sum(non_zero_mask)} ({sign_preserved/np.sum(non_zero_mask)*100:.2f}%)")
    
    logger.info("Realistic tensor test completed.")
    return True

if __name__ == "__main__":
    test_bf16_to_fp16_conversion()
    test_real_world_scenario()