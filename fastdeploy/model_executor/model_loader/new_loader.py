"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    get_all_safetensors,
    measure_time,
    safetensors_weights_iterator,
)
from fastdeploy.model_executor.load_weight_utils_cc70_compat import (
    safe_bf16_to_fp16_tensor,
)
from fastdeploy.config import get_cuda_compute_capability
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.platforms import current_platform


class NewModelLoader(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.synchronize()

    @measure_time
    def load_weights(self, model, fd_config: FDConfig) -> None:
        _, safetensor_files = get_all_safetensors(fd_config.model_config.model)
        
        # Check if we need CC70 compatibility
        compute_capability = get_cuda_compute_capability()
        if compute_capability >= 70 and compute_capability < 80:
            logger.info(f"Using CC70 compatible weight loading for compute capability {compute_capability}")
            # Create a wrapper iterator that handles bf16 to fp16 conversion
            def cc70_compat_weights_iterator(safetensor_files):
                for name, weight in safetensors_weights_iterator(safetensor_files):
                    # Convert numpy array to tensor for processing
                    if isinstance(weight, np.ndarray) and weight.dtype == np.dtype('bfloat16'):
                        logger.info(f"Converting bf16 weight '{name}' to fp16 for CC70 compatibility")
                        weight = safe_bf16_to_fp16_tensor(weight)
                    yield name, weight
            
            weights_iterator = cc70_compat_weights_iterator(safetensor_files)
        else:
            weights_iterator = safetensors_weights_iterator(safetensor_files)
        
        model.load_weights(weights_iterator)
        self.clean_memory_fragments()

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        logger.info(f"Starting to load model {architectures}")

        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

            architectures = architectures + "RL"

        model_cls = ModelRegistry.get_class(architectures)
        model = model_cls(fd_config)

        model.eval()

        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model

        self.load_weights(model, fd_config)
        return model
