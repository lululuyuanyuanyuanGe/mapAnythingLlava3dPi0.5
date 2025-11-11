# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import sys
import importlib
import torch
import torch.utils.checkpoint
from torch import nn
from torch.linalg import inv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers import AutoModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from .configuration_spatialvla import SpatialVLAConfig
from .modeling_gemma2 import Gemma2ForCausalLM
from .modeling_mapanything import MapAnythingWrapper

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

logger = logging.get_logger(__name__)

@dataclass
class SpatialVLACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

class SpatialVLAMultiModalProjector(nn.Module):
    def __init__(self, config: SpatialVLAConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class SpatialVLAPreTrainedModel(PreTrainedModel):
    config_class = SpatialVLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SpatialVLAMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class SpatialVLAForConditionalGeneration(SpatialVLAPreTrainedModel, GenerationMixin):
    def __init__(self, config: SpatialVLAConfig, vision_model=None, projector_model=None, language_model=None):
        super().__init__(config)

        self.vision_tower = vision_model or AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = projector_model or SpatialVLAMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        if getattr(config, "use_llava3d", False):
            self.language_model = LLaVA3DForCausalLM(config.text_config)
        else:
            self.language_model = Gemma2ForCausalLM(config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model
        self.geometric_model = MapAnythingWrapper(config)
        # This is the first MLP that map the mapanything output(1024 dims) to the siglip ourput(1052 dims)
        self.geometric_projector = nn.Linear(self.map_anything.config.hidden_size, self.vision_tower.config.hidden_size)
        self.fusion_projector = nn.Linear(self.vision_tower.config.hidden_size * 2, self.language_model.config.hidden_size)

        # monkey-patching
        if self.config.mm_projector_type == 'llava_llama':
            def _spatialvla_encode_images(self, pixel_values):
                # This is a monkey-patched method to replace the original `encode_images`
                # in LlavaLlamaForCausalLM. It uses the fused features from SpatialVLA
                # instead of recalculating them.
                # `pixel_values` here are actually the pre-computed fused features.
                return pixel_values

            import types
            self.language_model.encode_images = types.MethodType(_spatialvla_encode_images, self.language_model)

            original_prepare_inputs_for_generation = self.language_model.prepare_inputs_for_generation

            def _spatialvla_prepare_inputs_for_generation(self, *args, **kwargs):
                # This is a monkey-patched method to hijack the original `prepare_inputs_for_generation`
                # in LlavaLlamaForCausalLM. It allows us to pass `inputs_embeds` to `generate`.
                model_inputs = original_prepare_inputs_for_generation(*args, **kwargs)
                # The `inputs_embeds` are not passed to the original method, but we add them back here.
                if "inputs_embeds" in kwargs:
                    model_inputs["inputs_embeds"] = kwargs["inputs_embeds"]
                return model_inputs

            self.language_model.prepare_inputs_for_generation = types.MethodType(_spatialvla_prepare_inputs_for_generation, self.language_model)




    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = model_embeds.weight.shape[0]
        self.config.text_config.vocab_size = self.vocab_size = self.config._vocab_size = vocab_size
        self.tie_weights()
        return model_embeds
    
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_ids=None,
        inputs_embeds=None,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device)
        if sequence_length != 1:
            if is_training: causal_mask = torch.triu(causal_mask, diagonal=1)
            else: causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0)
        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        # Process through the semantic branch
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.last_hidden_state

        # Process through the geometric branch
        geometric_features = self.geometric_model(pixel_values, intrinsics)
        projected_geometric_features = self.geometric_projector(geometric_features)

        # Fusion
        fused_features = torch.cat([selected_image_feature, projected_geometric_features], dim=1)
        projected_fused_features = self.fusion_projector(fused_features)

        # Final projection and normalization
        image_features = self.multi_modal_projector(projected_fused_features)
        image_features = self.norm(image_features)

        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        actions: Optional[torch.FloatTensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, SpatialVLACausalLMOutputWithPast]:

        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None
        
        if inputs_embeds is None: inputs_embeds = self.get_input_embeddings()(input_ids).clone() # avoid checkpint grad True

        if self.config.use_spatial_token:
            spatial_selected = (input_ids >= self.config.action_token_begin_idx) & (input_ids < self.config.action_token_begin_idx + self.config.spatial_token_num)
            inputs_embeds[spatial_selected] = inputs_embeds[spatial_selected] * 0.0 + self.spatial_embed_tokens(input_ids[spatial_selected] - self.config.action_token_begin_idx)

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if getattr(self.config, "use_llava3d", False):
            # LLaVA-3D uses 0-indexed position_ids
            if position_ids is None:
                # Get device and sequence length correctly
                device = inputs_embeds.device
                seq_length = inputs_embeds.shape[1]
                batch_size = inputs_embeds.shape[0]
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # merge
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, intrinsic)
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None:
            labels = torch.where(input_ids == self.pad_token_id, IGNORE_INDEX, labels)

        # 在modeling_spatialvla.py中修改调用language_model的部分
        if isinstance(self.language_model, LLaVA3DForCausalLM):
            # 对于LLaVA3D模型，使用布尔类型的attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            
            outputs = self.language_model(
                attention_mask=attention_mask,  # 使用布尔类型的attention_mask
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # 移除LLaVA3D不支持的参数
                # cache_position=cache_position,
                # num_logits_to_keep=num_logits_to_keep,
            )
        else:
            # 对于其他模型，继续使用原来的causal_mask
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training
            )
            # 1105 zzq language_model必须传入input_embeds或者inputs_ids,
            # 如果没有传入inputs_embeds，则必须传入input_ids,这个input_ids和输入的#input_ids不同的是，
            # 输入的input_ids是原始的input_ids，而这个input_ids是prepare_inputs_labels_for_multimodal生成的input_ids，包含多模态信息
           
           
            # 这里应该传入input_embeds
            outputs = self.language_model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )

        logits = outputs.logits
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SpatialVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    # AR inference
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        intrinsic=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            if getattr(self.config, "use_llava3d", False):
            # LLaVA-3D不使用_update_causal_mask方法，直接使用attention_mask
                causal_mask = attention_mask
            
            # 导入LLaVA-3D的特殊标记常量
            from model.modeling_llava3d import LLAVA3D_IMAGE_TOKEN_INDEX, LLAVA3D_IGNORE_INDEX
            from ..LLaVA_3D.llava.constants import LOC_TOKEN_INDEX as LLAVA3D_LOC_TOKEN_INDEX
            
            # 如果使用LLaVA-3D，需要转换图像token索引
            # SpatialVLA: IMAGE_TOKEN_INDEX = 256000
            # LLaVA-3D: IMAGE_TOKEN_INDEX = -200
            if input_ids is not None and self.config.image_token_index != LLAVA3D_IMAGE_TOKEN_INDEX:
                # 创建临时副本以避免修改原始input_ids
                input_ids_for_llava3d = input_ids.clone()
                
                # 将SpatialVLA的图像token索引转换为LLaVA-3D的图像token索引
                # SpatialVLA: 256000, LLaVA-3D: -200 (IMAGE_TOKEN_INDEX)
                input_ids_for_llava3d[input_ids == self.config.image_token_index] = LLAVA3D_IMAGE_TOKEN_INDEX
                
                # 处理IGNORE_INDEX (LLaVA-3D中为-100)
                # 如果SpatialVLA中有不同的ignore_index值，也需要转换
                if hasattr(self.config, "ignore_index") and self.config.ignore_index != LLAVA3D_IGNORE_INDEX:
                    input_ids_for_llava3d[input_ids == self.config.ignore_index] = LLAVA3D_IGNORE_INDEX
                
                # 处理LOC_TOKEN_INDEX (LLaVA-3D中为-300)
                # 如果SpatialVLA中有类似的位置token，也需要转换
                if hasattr(self.config, "loc_token_index"):
                    input_ids_for_llava3d[input_ids == self.config.loc_token_index] = LLAVA3D_LOC_TOKEN_INDEX
                
                # 更新model_inputs中的input_ids
                model_inputs["input_ids"] = input_ids_for_llava3d
        else:
            causal_mask = self._update_causal_mask(attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training)
            model_inputs["attention_mask"] = causal_mask
        # 将 inputs_embeds 保留在返回字典中，供本模型 forward 使用以完成图像特征注入
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        model_inputs["intrinsic"] = intrinsic
        return model_inputs

    @torch.no_grad()
    def predict_action(
        self,
        model_inputs,
    ) -> torch.Tensor:
        model_inputs = model_inputs.to(torch.bfloat16).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        return generation_outputs[:,input_len:]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        if model.config.use_spatial_token: 
            model.language_model.model.embed_tokens.weight.data[-model.config.spatial_token_num:] = model.spatial_embed_tokens.weight.data
        return model