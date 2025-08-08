#!/usr/bin/env python3
"""
Test script to verify the improved bf16 to fp16 conversion fix.
This script tests the conversion with sample data to ensure output quality.
"""

import numpy as np
import paddle
import sys
import os

# Add the fastdeploy path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fastdeploy'))

def create_test_data():
    """Create test data that represents typical model weight distributions"""
    np.random.seed(42)
    
    # Create different types of test data that might be found in model weights
    test_cases = []
    
    # Case 1: Normal distribution (typical weights)
    normal_weights = np.random.normal(0, 0.1, 1000).astype(np.bfloat16)
    test_cases.append(("Normal distribution", normal_weights))
    
    # Case 2: Weights with some large values (potential overflow)
    mixed_weights = np.random.normal(0, 1.0, 1000).astype(np.bfloat16)
    # Add some large values that would overflow fp16
    mixed_weights[10:15] = np.array([70000, -80000, 90000, -75000, 85000], dtype=np.bfloat16)
    test_cases.append(("Mixed with large values", mixed_weights))
    
    # Case 3: Small values (denormals)
    small_weights = np.random.normal(0, 1e-6, 1000).astype(np.bfloat16)
    test_cases.append(("Small values (denormals)", small_weights))
    
    # Case 4: Mixed distribution with various edge cases
    edge_weights = np.random.normal(0, 0.5, 1000).astype(np.bfloat16)
    edge_weights[20:25] = np.array([np.inf, -np.inf, np.nan, 65520, -65520], dtype=np.bfloat16)
    test_cases.append(("Edge cases", edge_weights))
    
    return test_cases

def test_conversion_function(test_cases, conversion_func, func_name):
    """Test a conversion function with various test cases"""
    print(f"\n=== Testing {func_name} ===")
    
    results = []
    
    for case_name, bf16_data in test_cases:
        print(f"\nTesting: {case_name}")
        
        # Convert to paddle tensor
        bf16_tensor = paddle.to_tensor(bf16_data)
        
        # Apply conversion
        try:
            fp16_tensor = conversion_func(bf16_tensor)
            
            # Convert back to numpy for analysis
            fp16_data = fp16_tensor.numpy()
            bf32_data = bf16_data.astype(np.float32)  # Original in float32 for comparison
            
            # Calculate statistics
            original_mean = np.mean(bf32_data)
            original_std = np.std(bf32_data)
            converted_mean = np.mean(fp16_data)
            converted_std = np.std(fp16_data)
            
            # Calculate relative error
            relative_error = np.abs((bf32_data - fp16_data.astype(np.float32)) / (bf32_data + 1e-8))
            max_error = np.max(relative_error)
            mean_error = np.mean(relative_error)
            
            # Count overflow/underflow cases
            overflow_count = np.sum(np.abs(bf32_data) > 65504)
            underflow_count = np.sum(np.abs(bf32_data) < 6.103515625e-05)
            
            result = {
                'case_name': case_name,
                'original_mean': original_mean,
                'original_std': original_std,
                'converted_mean': converted_mean,
                'converted_std': converted_std,
                'max_error': max_error,
                'mean_error': mean_error,
                'overflow_count': overflow_count,
                'underflow_count': underflow_count,
                'success': True
            }
            
            print(f"  Original: mean={original_mean:.6f}, std={original_std:.6f}")
            print(f"  Converted: mean={converted_mean:.6f}, std={converted_std:.6f}")
            print(f"  Max relative error: {max_error:.6f}")
            print(f"  Mean relative error: {mean_error:.6f}")
            print(f"  Overflow cases: {overflow_count}, Underflow cases: {underflow_count}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                'case_name': case_name,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    return results

def main():
    """Main test function"""
    print("Testing BF16 to FP16 conversion improvements")
    print("=" * 50)
    
    # Create test data
    test_cases = create_test_data()
    
    # Test the new conversion function
    try:
        from fastdeploy.model_executor.load_weight_utils_cc70_compat import safe_bf16_to_fp16_tensor
        
        new_results = test_conversion_function(test_cases, safe_bf16_to_fp16_tensor, "New Improved Conversion")
        
        # Analyze results
        print("\n=== Results Analysis ===")
        successful_cases = [r for r in new_results if r['success']]
        failed_cases = [r for r in new_results if not r['success']]
        
        print(f"Successful conversions: {len(successful_cases)}/{len(new_results)}")
        if failed_cases:
            print(f"Failed conversions: {len(failed_cases)}")
            for case in failed_cases:
                print(f"  - {case['case_name']}: {case['error']}")
        
        if successful_cases:
            print("\nConversion quality metrics:")
            total_max_error = max(r['max_error'] for r in successful_cases)
            total_mean_error = np.mean([r['mean_error'] for r in successful_cases])
            
            print(f"  Maximum relative error across all cases: {total_max_error:.6f}")
            print(f"  Average mean relative error: {total_mean_error:.6f}")
            
            # Check if errors are within acceptable bounds
            if total_max_error < 0.1:  # 10% error threshold
                print("  ✓ Error rates are within acceptable bounds")
            else:
                print("  ⚠ Error rates may be too high for some applications")
            
            # Check statistical preservation
            mean_drifts = [abs(r['converted_mean'] - r['original_mean']) / (abs(r['original_mean']) + 1e-8) 
                          for r in successful_cases]
            std_drifts = [abs(r['converted_std'] - r['original_std']) / (abs(r['original_std']) + 1e-8) 
                         for r in successful_cases]
            
            avg_mean_drift = np.mean(mean_drifts)
            avg_std_drift = np.mean(std_drifts)
            
            print(f"  Average mean drift: {avg_mean_drift:.6f}")
            print(f"  Average std drift: {avg_std_drift:.6f}")
            
            if avg_mean_drift < 0.05 and avg_std_drift < 0.05:  # 5% drift threshold
                print("  ✓ Statistical properties are well preserved")
            else:
                print("  ⚠ Statistical drift may be significant")
        
    except ImportError as e:
        print(f"Could not import conversion function: {e}")
        print("Make sure the fastdeploy module is properly installed")
        return 1
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())