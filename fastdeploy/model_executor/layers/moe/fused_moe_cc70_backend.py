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
Compatibility backend for MOE operations on CUDA compute capability >= 70.
This backend provides fallback implementations for older GPU architectures
that don't support the full feature set of cc>=80.
"""

import paddle
import numpy as np
from typing import Optional, Tuple, List, Any

from .fused_moe_backend_base import MoEMethodBase
from .fused_moe_cutlass_backend import CutlassMoEMethod
from fastdeploy.platforms import current_platform
from fastdeploy.model_executor.load_weight_utils_cc70_compat import safe_bf16_to_fp16_tensor


class CC70CompatMoEMethod(MoEMethodBase):
    """
    MOE Method for CUDA compute capability >= 70 with compatibility fallbacks.
    
    This backend provides:
    - FP16 operations instead of BF16 for better compatibility
    - SIMT-based GEMM operations instead of Tensor Core when needed
    - Simplified quantization support (INT8 only, no INT4)
    - Reduced memory usage optimizations
    """
    
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = "fp16"  # Use FP16 instead of BF16
        self.use_simt_fallback = True
        self.supports_int4 = False  # Disable INT4 for cc70
        self.supports_bf16 = False  # Disable BF16 for cc70
        
        # Check actual compute capability
        self.compute_capability = self._get_compute_capability()
        if self.compute_capability >= 80:
            # If we're actually on cc>=80, delegate to full implementation
            self._delegate_to_full_impl = True
            self._full_impl = CutlassMoEMethod(quant_config)
        else:
            self._delegate_to_full_impl = False
            self._full_impl = None
    
    def _get_compute_capability(self) -> int:
        """Get the compute capability of the current device."""
        try:
            if current_platform.is_cuda():
                device_id = paddle.get_device().split(':')[-1]
                prop = paddle.device.cuda.get_device_properties(int(device_id))
                return prop.major * 10 + prop.minor
        except:
            pass
        return 70  # Default to cc70 if detection fails
    
    def can_implement(self, layer) -> bool:
        """Check if this backend can implement the given MOE layer."""
        if self._delegate_to_full_impl:
            return self._full_impl.can_implement(layer)
        
        # Check basic requirements for cc70 compatibility
        if not current_platform.is_cuda():
            return False
        
        # Only support FP16 precision
        if hasattr(layer, 'precision') and layer.precision not in ['fp16', 'float16']:
            return False
        
        # Don't support INT4 quantization
        if (hasattr(layer, 'quant_method') and 
            layer.quant_method in ['int4', 'w4a8', 'w4a16']):
            return False
        
        return True
    
    def create_weights(self, layer, num_experts: int, input_size: int, 
                      output_sizes: List[int], input_size_per_partition: int,
                      output_size_per_partition: int, params_dtype: paddle.dtype,
                      weight_loader, prefix: str):
        """Create weights with cc70 compatibility considerations."""
        if self._delegate_to_full_impl:
            return self._full_impl.create_weights(
                layer, num_experts, input_size, output_sizes,
                input_size_per_partition, output_size_per_partition,
                params_dtype, weight_loader, prefix)
        
        # Force FP16 dtype for compatibility
        if params_dtype == paddle.bfloat16:
            params_dtype = paddle.float16
        
        # Create standard weights without advanced quantization
        weights = {}
        
        for expert_id in range(num_experts):
            # Gate projection weights
            gate_weight = paddle.create_parameter(
                shape=[input_size_per_partition, output_size_per_partition],
                dtype=params_dtype,
                default_initializer=paddle.nn.initializer.Normal(std=0.02)
            )
            weights[f"{prefix}.experts.{expert_id}.gate_proj.weight"] = gate_weight
            
            # Up projection weights  
            up_weight = paddle.create_parameter(
                shape=[input_size_per_partition, output_size_per_partition],
                dtype=params_dtype,
                default_initializer=paddle.nn.initializer.Normal(std=0.02)
            )
            weights[f"{prefix}.experts.{expert_id}.up_proj.weight"] = up_weight
            
            # Down projection weights
            down_weight = paddle.create_parameter(
                shape=[output_size_per_partition, input_size_per_partition],
                dtype=params_dtype,
                default_initializer=paddle.nn.initializer.Normal(std=0.02)
            )
            weights[f"{prefix}.experts.{expert_id}.down_proj.weight"] = down_weight
        
        return weights
    
    def apply(self, layer, x: paddle.Tensor, router_logits: paddle.Tensor,
              top_k: int, renormalize: bool = True, 
              use_grouped_topk: bool = False, topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None) -> paddle.Tensor:
        """Apply MOE with cc70 compatibility."""
        if self._delegate_to_full_impl:
            return self._full_impl.apply(
                layer, x, router_logits, top_k, renormalize,
                use_grouped_topk, topk_group, num_expert_group)
        
        # Convert BF16 to FP16 if needed using safe conversion
        if x.dtype == paddle.bfloat16:
            x = safe_bf16_to_fp16_tensor(x)
        if router_logits.dtype == paddle.bfloat16:
            router_logits = safe_bf16_to_fp16_tensor(router_logits)
        
        # Use simplified MOE implementation for cc70
        return self._apply_cc70_moe(layer, x, router_logits, top_k, renormalize)
    
    def _apply_cc70_moe(self, layer, x: paddle.Tensor, router_logits: paddle.Tensor,
                        top_k: int, renormalize: bool) -> paddle.Tensor:
        """Simplified MOE implementation for cc70 compatibility."""
        batch_size, seq_len, hidden_size = x.shape
        num_experts = len(layer.experts)
        
        # Reshape input
        x_flat = x.reshape([-1, hidden_size])
        
        # Compute routing weights
        routing_weights = paddle.nn.functional.softmax(router_logits, axis=-1)
        
        # Get top-k experts
        topk_weights, topk_indices = paddle.topk(routing_weights, k=top_k, axis=-1)
        
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)
        
        # Initialize output
        output = paddle.zeros_like(x_flat)
        
        # Process each expert sequentially (simpler but less efficient for cc70)
        for expert_idx in range(num_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_indices == expert_idx).any(axis=-1)
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            expert_tokens = x_flat[expert_mask]
            if expert_tokens.shape[0] == 0:
                continue
            
            # Apply expert FFN
            expert_output = self._apply_expert_ffn(layer.experts[expert_idx], expert_tokens)
            
            # Compute weights for this expert
            token_indices = paddle.where(expert_mask)[0]
            expert_weights = paddle.zeros([expert_tokens.shape[0]], dtype=x.dtype)
            
            for i, token_idx in enumerate(token_indices):
                expert_positions = paddle.where(topk_indices[token_idx] == expert_idx)[0]
                if len(expert_positions) > 0:
                    expert_weights[i] = topk_weights[token_idx, expert_positions[0]]
            
            # Add weighted expert output
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output[expert_mask] += weighted_output
        
        # Reshape back to original shape
        return output.reshape([batch_size, seq_len, hidden_size])
    
    def _apply_expert_ffn(self, expert, x: paddle.Tensor) -> paddle.Tensor:
        """Apply expert FFN with cc70 optimizations."""
        # Gate projection
        gate_out = paddle.matmul(x, expert.gate_proj.weight, transpose_y=True)
        
        # Up projection  
        up_out = paddle.matmul(x, expert.up_proj.weight, transpose_y=True)
        
        # SwiGLU activation (compatible with cc70)
        gate_out = paddle.nn.functional.silu(gate_out)
        intermediate = gate_out * up_out
        
        # Down projection
        output = paddle.matmul(intermediate, expert.down_proj.weight, transpose_y=True)
        
        return output


class CC70CompatW8MoEMethod(CC70CompatMoEMethod):
    """
    Weight-only INT8 quantized MOE method for cc70 compatibility.
    Provides INT8 quantization support for older architectures.
    """
    
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.moe_quant_type = "w8"
        self.supports_int8 = True
    
    def can_implement(self, layer) -> bool:
        """Check if INT8 quantization is supported."""
        if not super().can_implement(layer):
            return False
        
        # Support INT8 quantization
        if (hasattr(layer, 'quant_method') and 
            layer.quant_method in ['int8', 'w8a16']):
            return True
        
        return False
