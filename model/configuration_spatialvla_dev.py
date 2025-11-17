# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
from LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig

logger = logging.get_logger(__name__)

class SpatialVLAConfig(PretrainedConfig):
    model_type = "spatialvla"
    # 仅声明始终存在的子配置，避免在自动设置注意力实现时访问 None
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        vision_zoe_config=None,
        action_token_begin_idx=None,
        spatial_token_num=259,
        use_spatial_token=False,
        ego3d_patch_reso=4,
        n_freqs=8,
        use_vision_zoe=False,
        use_llava3d=True,  # 新增参数
        llava3d_model_type="llama",  # 新增参数
        **kwargs,
    ):
        # 根据 use_llava3d 自动设置 ignore_index
        if ignore_index is None:
            if use_llava3d:
                self._ignore_index = LLAVA3D_IGNORE_INDEX  # LLaVA3D 推荐值
            else:
                self._ignore_index = -100  # 与原版/HF约定一致
        else:
            self._ignore_index = ignore_index

            
        self.image_token_index = image_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )
        if use_llava3d:
            self.use_llava3d = True
            self.llava3d_model_type = llava3d_model_type
            # 可以根据LLaVA-3D的需求调整text_config
            if isinstance(text_config, dict):
                text_config["model_type"] = text_config.get("model_type", llava3d_model_type)
                self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            elif text_config is None:
                # 使用LLaVA-3D默认配置
                if llava3d_model_type == "llama":
                    self.text_config = CONFIG_MAPPING["llama"](
                        hidden_size=4096,
                        num_hidden_layers=32,
                        intermediate_size=11008,
                        num_attention_heads=32,
                        vocab_size=vocab_size,
                    )
                elif llava3d_model_type == "mistral":
                    self.text_config = CONFIG_MAPPING["mistral"](
                        hidden_size=4096,
                        num_hidden_layers=32,
                        intermediate_size=14336,
                        num_attention_heads=32,
                        vocab_size=vocab_size,
                    )
            else:
                # 传入的是一个已构建好的配置对象
                self.text_config = text_config
        else:
            if isinstance(text_config, dict):
                text_config["model_type"] = text_config.get("model_type", "gemma2")
                self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            elif text_config is None:
                self.text_config = CONFIG_MAPPING["gemma2"](
                    hidden_size=2048,
                    num_hidden_layers=18,
                    intermediate_size=16384,
                    num_attention_heads=8,
                    num_key_value_heads=1,
                    is_encoder_decoder=False,
                    vocab_size=vocab_size,
                )
            else:
                self.text_config = text_config
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

        # vision zoe config
        self.vision_zoe_config = vision_zoe_config
        if use_vision_zoe:
            if isinstance(self.vision_zoe_config, dict):
                vision_zoe_config["model_type"] = vision_zoe_config.get("model_type", "zoedepth")
                self.vision_zoe_config = CONFIG_MAPPING[vision_zoe_config["model_type"]](**vision_zoe_config)
            elif self.vision_zoe_config is None:
                # Provide a safe default ZoeDepth config when depth is enabled but not provided
                try:
                    self.vision_zoe_config = CONFIG_MAPPING["zoedepth"]()
                except Exception:
                    # If zoedepth is unavailable, gracefully disable depth usage
                    self.use_vision_zoe = False
            else:
                # Already a config object, keep as is
                pass
        else:
            # Depth not used; leave as provided (likely None)
            pass
        
        # additional attributes
        self.action_token_begin_idx = action_token_begin_idx
        self.spatial_token_num = spatial_token_num
        self.use_spatial_token = use_spatial_token
        self.ego3d_patch_reso = ego3d_patch_reso
        self.n_freqs = n_freqs
        self.use_vision_zoe = use_vision_zoe

        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output