# 在SpatialVLA/model/目录下创建新文件
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Union

# 导入LLaVA-3D相关模块
import sys
sys.path.append('/Users/bazinga/Downloads/LLaVA-3D-main')
from ..LLaVA_3D.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from ..LLaVA_3D.llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from ..LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
from ..LLaVA_3D.llava.constants import IMAGE_TOKEN_INDEX as LLAVA3D_IMAGE_TOKEN_INDEX

# 常量值转换 (SpatialVLA与LLaVA-3D的常量值不同)
# SpatialVLA: IMAGE_TOKEN_INDEX = 256000
# LLaVA-3D: IMAGE_TOKEN_INDEX = -200

# 创建LLaVA-3D语言模型适配器
class LLaVA3DForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        from ..LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
        self.ignore_index = LLAVA3D_IGNORE_INDEX
        # 根据配置选择合适的LLaVA-3D语言模型
        model_type = getattr(config, "llava3d_model_type", "llama")
        if model_type == "llama":
            self.model = LlavaLlamaForCausalLM(config)
        elif model_type == "mistral":
            self.model = LlavaMistralForCausalLM(config)
        else:
            raise ValueError(f"Unsupported LLaVA-3D model type: {model_type}")
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)