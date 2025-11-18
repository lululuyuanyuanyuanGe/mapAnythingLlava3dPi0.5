#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试新架构下SpatialVLA与LLaVA-3D的集成功能
主要测试：
1. 灵活的视觉模型加载（官方模型 vs. 从SpatialVLA提取）
2. 统一的特殊标记转换逻辑
3. 端到端推理流程
"""

import argparse
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoTokenizer

# Force protobuf to use pure-Python implementation to avoid sentencepiece pb2 C++ issues in tests
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# 导入重构后的SpatialVLA模块
from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration
from model.configuration_spatialvla_dev import SpatialVLAConfig

# 从LLaVA-3D导入常量
from LLaVA_3D.llava.constants import (
    IMAGE_TOKEN_INDEX as LLAVA3D_IMAGE_TOKEN_INDEX
)

# --- Mocking Zone for Fast Unit Tests ---
class MockLLaVAModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
    def get_input_embeddings(self):
        return self.embed_tokens
    def forward(self, inputs_embeds, **kwargs):
        batch, seq, hidden = inputs_embeds.shape
        output = MagicMock()
        output.loss = torch.tensor(0.5, device=inputs_embeds.device)
        output.logits = torch.randn(batch, seq, self.vocab_size, device=inputs_embeds.device)
        output.past_key_values = None
        return output
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

class MockVisionTower(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, pixel_values, **kwargs):
        batch_size = pixel_values.shape[0]
        # config may be the full SpatialVLAConfig; prefer text_config.num_image_tokens
        vision_cfg = getattr(self.config, "vision_config", self.config)
        hidden_size = getattr(vision_cfg, "hidden_size", 1152)
        text_cfg = getattr(self.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_image_tokens"):
            num_patches = int(text_cfg.num_image_tokens)
        else:
            img_size = getattr(vision_cfg, "image_size", 224)
            patch_size = getattr(vision_cfg, "patch_size", 16)
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            num_patches = (img_size // patch_size) ** 2
        output = MagicMock()
        output.last_hidden_state = torch.randn(
            batch_size,
            num_patches,
            hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        return output
# --- End Mocking Zone ---


class TestNewLLaVA3DIntegration(unittest.TestCase):
    """测试新架构下SpatialVLA与LLaVA-3D的集成功能"""

    @classmethod
    @patch('model.modeling_spatialvla_dev.LLaVA3DForCausalLMV2.from_pretrained', side_effect=lambda path, **kwargs: MockLLaVAModel(AutoConfig.from_pretrained(path, **kwargs)))
    @patch('model.modeling_spatialvla_dev.SpatialVLAForConditionalGeneration._init_vision_tower', autospec=True)
    def setUpClass(cls, mock_init_vt, mock_llava):
        """设置测试环境，使用Mock模型进行快速单元测试"""
        parser = argparse.ArgumentParser("SpatialVLA新架构集成测试")
        parser.add_argument("--language_model_path", type=str, default="NousResearch/Llama-2-7b-hf", help="语言模型路径 (用于配置)")
        parser.add_argument("--vision_model_path", type=str, default="google/siglip-base-patch16-224", help="独立的视觉模型路径")
        parser.add_argument("--spatialvla_model_path", type=str, default=None, help="旧SpatialVLA模型路径 (用于提取视觉权重)")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
        parser.add_argument("--image_path", type=str, default=None, help="测试图片路径")
        
        cls.args, _ = parser.parse_known_args(os.environ.get("TEST_ARGS", "").split())
        
        # 确保vision_model_path和spatialvla_model_path是互斥的
        if cls.args.vision_model_path and cls.args.spatialvla_model_path:
            raise ValueError("只能提供 --vision_model_path 或 --spatialvla_model_path 中的一个，不能同时提供。")

        dev_str = str(cls.args.device)
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        elif dev_str.lower() in ("gpu",) and torch.cuda.is_available():
            dev_str = "cuda:0"
        cls.device = torch.device(dev_str)

        # 根据传入参数动态构建配置
        config_kwargs = {
            "language_model_name_or_path": cls.args.language_model_path,
            "image_token_index": 256000,
            "ignore_index": -100,
        }
        
        if cls.args.spatialvla_model_path:
            print(f"配置模式：使用 SpatialVLA ({cls.args.spatialvla_model_path}) 的视觉权重。")
            # 基础模型骨架仍然是官方SigLIP
            config_kwargs["vision_model_name_or_path"] = "google/siglip-base-patch16-224"
            config_kwargs["vision_weight_source"] = "spatialvla_vision_only"
            config_kwargs["spatialvla_vision_pretrained_path"] = cls.args.spatialvla_model_path
        else:
            print(f"配置模式：使用独立的视觉模型 ({cls.args.vision_model_path})。")
            config_kwargs["vision_model_name_or_path"] = cls.args.vision_model_path
            
        cls.config = SpatialVLAConfig(**config_kwargs)
        # 使 _init_vision_tower 返回使用测试配置的 MockVisionTower
        mock_init_vt.return_value = MockVisionTower(cls.config)
        # 初始化模型
        cls.model = SpatialVLAForConditionalGeneration(cls.config)
        cls.model.to(cls.device)
        cls.model.eval()
        
        cls.prepare_test_data()
    
    @classmethod
    def prepare_test_data(cls):
        """准备测试数据：使用真实 Processor 构造输入，并注入占位符以测试转换"""
        cls.spatialvla_placeholder_idx = cls.config.image_token_index
        # 构造processor（需要image_seq_length、tokenizer）
        vision_path = cls.args.vision_model_path or "google/siglip-base-patch16-224"
        try:
            tokenizer = AutoTokenizer.from_pretrained(cls.args.language_model_path, use_fast=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(cls.args.language_model_path, use_fast=False)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
        img_processor = AutoImageProcessor.from_pretrained(vision_path)
        # 补充 image_seq_length 属性供 Processor 使用
        seq_len = int(getattr(cls.config.text_config, "num_image_tokens", 1))
        setattr(img_processor, "image_seq_length", seq_len)
        from model.processing_spatialvla_dev import SpatialVLAProcessor
        # 传入必要的processor配置
        statistics = {"default": {"action": {"q01": [0,0,0], "q99": [1,1,1], "mask": [1,1,1]}}}
        intrinsic_config = {"default": {"width": 224, "height": 224, "intrinsic": [[200,0,112],[0,200,112],[0,0,1]]}}
        # Minimal, explicit bin configuration for SpatialActionTokenizer
        action_config = {
            "num_bins": {
                "translation": {"theta_bins": 4, "phi_bins": 4, "r_bins": 4},
                "rotation": {"roll_bins": 4, "pitch_bins": 4, "yaw_bins": 4},
                "gripper": 2,
            },
            "use_spherical": False,
        }
        # Provide explicit uniform bin boundaries (len = bins + 1)
        bin_policy = {
            "translation": {
                "theta_bins": [0.0, 0.785398, 1.570796, 2.356194, 3.141593],
                "phi_bins": [-3.141593, -1.570796, 0.0, 1.570796, 3.141593],
                "r_bins": [0.0, 0.433013, 0.866025, 1.299038, 1.732051],
            },
            "rotation": {
                "roll_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "pitch_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "yaw_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
        }
        processor = SpatialVLAProcessor(
            image_processor=img_processor,
            tokenizer=tokenizer,
            statistics=statistics,
            bin_policy=bin_policy,
            intrinsic_config=intrinsic_config,
            action_config=action_config,
        )
        # 使用用户提供的真实图片路径（如果存在），否则创建一个占位图片
        img_path = cls.args.image_path
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new('RGB', (224, 224), color = 'blue')
        prompts = ["Describe the image.", "Describe again."]
        inputs = processor(
            images=[image, image],
            text=prompts,
            return_tensors="pt",
        )
        # 保存原始processor输出用于前向测试（不插入额外占位符）
        cls.inputs_forward = inputs
        # 构造一个仅用于转换测试的变体：在开头插入一个占位符 sentinel
        cls.real_image_token_id = int(inputs["image_token_id"]) if isinstance(inputs["image_token_id"], int) else int(inputs["image_token_id"]) 
        input_ids_conv = inputs["input_ids"].clone()
        input_ids_conv[0, 0] = cls.spatialvla_placeholder_idx
        cls.inputs_convert = {**inputs, "input_ids": input_ids_conv}
    
    def test_token_conversion_is_unconditional(self):
        """测试特殊标记转换逻辑"""
        print("\n测试特殊标记转换逻辑...")
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=self.inputs_convert["input_ids"].clone(),
            image_token_index=self.inputs_convert["image_token_index"],
            image_token_id=self.inputs_convert["image_token_id"],
        )
        converted_input_ids = model_inputs["input_ids"]
        original_placeholder_pos = (self.inputs_convert["input_ids"] == self.spatialvla_placeholder_idx)
        original_real_id_pos = (self.inputs_convert["input_ids"] == self.inputs_convert["image_token_id"]) 

        self.assertTrue(torch.all(converted_input_ids[original_placeholder_pos] == LLAVA3D_IMAGE_TOKEN_INDEX))
        self.assertTrue(torch.all(converted_input_ids[original_real_id_pos] == LLAVA3D_IMAGE_TOKEN_INDEX))
        print("✓ 占位符和真实图像Token ID均被正确转换为LLaVA哨兵值")

    def test_forward_pass_with_pixel_values(self):
        """测试带有图像输入的模型前向传播"""
        print("\n测试模型前向传播...")
        try:
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in self.inputs_forward.items()}
            outputs = self.model(
                **inputs,
                return_dict=True,
            )
            self.assertIsNotNone(outputs.loss)
            self.assertIsNotNone(outputs.logits)
            print("✓ 模型前向传播（带图像）测试通过")
        except Exception as e:
            self.fail(f"模型前向传播（带图像）测试失败: {e}")

    def test_integration_end_to_end(self):
        """端到端集成测试，使用真实模型（如果路径被提供）"""
        # 仅在提供整合好的 VLA 目录时才进行端到端测试
        if not (self.args.spatialvla_model_path and os.path.isdir(self.args.spatialvla_model_path)):
            print("\n⚠ 跳过端到端测试：需要一个整合好的 VLA 目录 (--spatialvla_model_path)")
            return

        print("\n执行端到端集成测试...")
        
        try:
            # 仅使用整合好的 VLA 目录进行端到端加载
            model = SpatialVLAForConditionalGeneration.from_pretrained(
                self.args.spatialvla_model_path,
            ).to(self.device)
            
            # 假设processor与整合好的VLA模型相关联
            processor = AutoProcessor.from_pretrained(self.args.spatialvla_model_path, trust_remote_code=True)

            print(f"[E2E] 模型加载成功。语言模型: {model.config.language_model_name_or_path}")
            print(f"[E2E] 视觉模型来源: {model.config.vision_weight_source}, 路径: {model.config.vision_model_name_or_path}")
            
            # 2. 准备输入
            img_path = self.args.image_path
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.new('RGB', (224, 224), color = 'blue')
            prompt = "Describe the image."
            inputs = processor(images=[image], text=prompt, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            print(f"[E2E] 输入准备成功，keys: {list(inputs.keys())}")
            
            # 3. 执行推理
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            
            # 4. 解码
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"[E2E] 生成的文本: {generated_text}")
            self.assertTrue(len(generated_text) > 0)
            
            print("✓ 端到端集成测试通过")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"端到端集成测试失败: {e}")

if __name__ == "__main__":
    # 解析所有已知参数，并将其打包到环境变量中
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--language_model_path", type=str)
    parser.add_argument("--vision_model_path", type=str)
    parser.add_argument("--spatialvla_model_path", type=str)
    parser.add_argument("--device", type=str)
    
    # 将 sys.argv 分为已知参数和 unittest 参数
    known_args, remaining_argv = parser.parse_known_args()
    
    # 将已知参数格式化为字符串，以便 setUpClass 解析
    test_args_list = []
    for key, value in vars(known_args).items():
        if value is not None:
            test_args_list.append(f"--{key}={value}")
    os.environ["TEST_ARGS"] = " ".join(test_args_list)
    
    # 运行 unittest，只传递它自己的参数
    unittest.main(argv=[sys.argv[0]] + remaining_argv)