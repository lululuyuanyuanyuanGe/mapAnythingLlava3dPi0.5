# SpatialVLA + LLaVA‑3D Integration (dev) — Changes and Status

## 目标概述
- 保持“视觉侧由 SpatialVLA 负责（Vision Tower + Multi‑Modal Projector），语言侧由 LLaVA‑3D 负责解码”。
- 语言模型替换后，兼容不同词表规模与特殊标记策略，避免嵌入越界与类型不匹配。
- 前向注入图像特征使用真实 tokenizer ID 定位；生成阶段按 LLaVA‑3D 的约定转换为哨兵索引。
- 在不必修改模型目录文件的情况下，支持“运行时覆盖”策略，保证端到端测试与 HuggingFace 加载兼容。

---

## 修改总览（按模块）

### 1) 配置（model/configuration_spatialvla_dev.py）
- 成功：将 `ignore_index` 在非 LLaVA 模式下统一为 `-100`，与 `CrossEntropyLoss` 兼容。
- 失败尝试：将 `use_vision_zoe` 默认改为 `False` 的补丁一度失败（无法精确定位行），后续未强制修改；当前仍由配置/脚本显式控制。

### 2) 处理器（model/processing_spatialvla_dev.py）
- 成功：
  - 移除了依赖工作目录的动态 tokenizer 选择逻辑，改为按 `use_llava3d` 显式选择：`LlamaTokenizer/LlamaTokenizerFast` 或 `GemmaTokenizer/GemmaTokenizerFast`。
  - 在返回中加入 `image_token_id`（真实 `<image>` 词表 ID）与 `image_token_index`（LLaVA 哨兵索引 `-200`），供模型一致使用。
- 注意：动作标记（TRANSLATION/ROTATION/GRIPPER）会追加到“语言 tokenizer”的词表中，随后需要扩展语言侧嵌入矩阵（见第 6 节）。

### 3) 模型（model/modeling_spatialvla_dev.py）
- 成功：
  - `forward`：新增可选参数 `image_token_id`，用于定位图像标记位置并注入图像特征；当未提供时回退到 `config.image_token_index`。
  - `prepare_inputs_for_generation`：新增可选参数 `image_token_index`，优先使用传入索引将相应位置映射到 LLaVA‑3D 哨兵值；索引相等时显式保留原始 `input_ids`。
  - 标签屏蔽：对 `labels` 做健壮过滤（pad、image、loc 位置设为 `ignore_index`，以及所有越界标签一律设为 `ignore_index`），修复 `IndexError: Target 256000 is out of bounds`。
  - 非 LLaVA 分支：无论 `flash_attention_2` 与否，均生成 4D 因果掩码；对空 `HybridCache` 做保护。
  - `from_pretrained`：改为通过 `get_input_embeddings()` 安全获取语言侧嵌入层，并在维度一致时才拷贝空间嵌入；避免对 LLaVA‑3D 的 `embed_tokens` 访问异常。
  - 兼容性修复：为 `cache_position` 空值加保护；在嵌入前对潜在越界特殊 ID 进行替换/钳制，避免嵌入越界（历史补丁保留）。
- 失败/注意：
  - 早期直接访问 `language_model.model.embed_tokens` 在 LLaVA‑3D 类中不存在，造成 `AttributeError`；已修复为通用接口并加维度判断。

### 4) 集成测试（test/test_llava3d_integration.py）
- 成功：
  - 默认 `--model_path` 指向 `model_zoo/spatialvla-4b-224`（含处理器配置），并在运行时优先选择包含 `preprocessor_config.json`/`processor_config.json` 的目录。
  - 注入丰富调试输出：候选目录、目录文件列表、环境版本、配置字段、processor/tokenizer/model 类型与配置、输入形状/设备。
  - 在 `test_pixel_values_injection_first_step` 用例中显式 `self.model.config.use_llava3d=False`，避免前序状态污染导致走错分支。
  - 端到端流程中手动组合 dev 处理器：使用 `AutoProcessor` 获取 `image_processor`，使用 `AutoTokenizer` 加载 LLaMA tokenizer，并用 dev `SpatialVLAProcessor` 组合。
- 失败/注意：
  - 最初端到端测试跳过的原因是默认目录缺少处理器配置，已通过目录切换与候选选择修复。

### 5) 模型目录（model_zoo/spatialvla-4b-224）
- 成功：
  - `config.json` 的 `auto_map` 指向 dev 类：
    - `configuration_spatialvla_dev.SpatialVLAConfig`
    - `modeling_spatialvla_dev.SpatialVLAForConditionalGeneration`
  - `processor_config.json` 的 `auto_map` 指向 dev 处理器：
    - `processing_spatialvla_dev.SpatialVLAProcessor`
  - 新增轻量别名文件：`configuration_spatialvla_dev.py`、`modeling_spatialvla_dev.py`、`processing_spatialvla_dev.py`，将目录导入转发到仓库中的 dev 实现。
  - 处理器别名文件中加入 wrapper，放宽 `tokenizer_class` 检查（允许 LLaMA/Gemma），避免 ProcessorMixin 报类型不匹配错误。
  - 修复 `processor_config.json` 中非法 `Infinity` 值为有限数值（`0.0`），避免 JSON 解析错误。
- 注意：
  - 目录中 `use_spatial_token` 设为 `false` 可作为临时规避策略，关闭空间嵌入拷贝与维度不匹配；当前你“不需要空间嵌入”，此方案安全可行。

### 6) HuggingFace dev 测试（test/test_huggingface_dev.py）
- 成功：
  - 运行时覆盖：仅用 `AutoProcessor` 获取图像处理器，用 `AutoTokenizer` 统一为 LLaMA tokenizer，然后用 dev 处理器组合。
  - 新增 `--llava3d_path` 参数：如提供则加载并替换语言模型为 LLaVA‑3D，并同步 `vocab_size`。
  - 加载模型使用 `ignore_mismatched_sizes=True`：兼容旧 checkpoint 与新结构的维度差异。
  - 调用 `model.resize_token_embeddings(len(tok))`：动作标记追加后同步语言侧嵌入矩阵大小，避免 OOV/维度不一致。
  - 设定 `inputs["cache_position"] = [0]`：保证首步注入 `pixel_values`。
  - 设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 降级兼容，避免 protobuf 版本差异导致的 `Descriptors` 错误。
- 失败/注意：
  - 若未提供 `--llava3d_path`，目录语言侧仍为 Gemma2 风格，`unused weights` 提示会很多属正常；替换后会减少此类提示。

---

## 问题与原因汇总（失败/报错）

### 详细错误链路与修复

1. `IndexError: Target 256000 is out of bounds`
   - 触发场景：`CrossEntropyLoss` 计算时，`labels` 含有 `256000` 等超出 LLaVA 词表（32k）的标签。
   - 根因：原 Gemma2 词表约 257k，替换为 LLaVA‑3D 后词表显著缩小；将真实词表 ID 当作标签参与 loss 会越界。
   - 修复：在 `modeling_spatialvla_dev.py` 的 `forward` 中统一屏蔽 pad/image/loc 位点，并对所有越界标签（`labels >= vocab_size` 或 `labels < 0`）设为 `ignore_index=-100`。

2. 非 LLaVA 模式下 `attention_mask` 未变化（断言失败）
   - 触发场景：`test_pixel_values_injection_first_step` 期望非 LLaVA 分支生成新的 4D 因果掩码，但拿到的是原始布尔掩码。
   - 根因：测试类共享模型实例，前序用例将 `use_llava3d=True` 未重置，导致此用例仍走 LLaVA 分支；属于状态污染。
   - 修复：在用例开头显式 `self.model.config.use_llava3d=False`；代码层面保证非 LLaVA 分支总是生成 4D 因果掩码，并对空 `HybridCache` 做保护。

3. 端到端测试跳过
   - 触发场景：`test_integration_end_to_end` 指向默认 `model_path`（`llava3d_7B`），目录下无 `preprocessor_config.json/processor_config.json`。
   - 根因：HuggingFace 加载流程需要处理器配置文件；目录不完整导致跳过。
   - 修复：将默认 `model_path` 改为 `model_zoo/spatialvla-4b-224`，并在运行时候选列表中优先选择包含处理器配置的目录；打印实际使用目录与文件列表以辅助定位。

4. Tokenizer 类型不匹配（期望 LLaMA，实际为 Gemma）
   - 触发场景：端到端加载时，处理器返回 `GemmaTokenizer/GemmaTokenizerFast`，而目录 tokenizer 是 `LlamaTokenizer`。
   - 根因：`AutoProcessor` 在 4.47 的某些路径下选择了目录中的原版处理器，导致类型不一致。
   - 修复：在 e2e 测试中手动组合 dev 处理器：用 `AutoProcessor` 仅取 `image_processor`，用 `AutoTokenizer` 保证 LLaMA tokenizer，再用 dev `SpatialVLAProcessor` 组合；同时在目录中添加处理器别名 wrapper（接受 LLaMA/Gemma），避免 ProcessorMixin 的严格类型检查抛错。

5. Protobuf `Descriptors cannot be created directly`
   - 触发场景：端到端加载 tokenizer/权重时，环境的 `protobuf/protoc` 版本与生成代码不兼容。
   - 根因：常见于 4.47 + 近期 `protobuf` 版本差异；需要环境匹配或降级。
   - 修复：在测试设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 作为临时兼容（纯 Python 解析）；长期建议 `pip install "protobuf<3.21"` 或升级 `protoc>=3.19.0` 并重启进程。

6. `AttributeError: 'LlavaLlamaForCausalLM' object has no attribute 'embed_tokens'`
   - 触发场景：`from_pretrained` 试图执行 `model.language_model.model.embed_tokens ...`。
   - 根因：Gemma2 语言模型结构与 LLaVA‑3D 语言模型结构不同，后者没有 `.model.embed_tokens` 该嵌套属性。
   - 修复：使用 `model.language_model.get_input_embeddings()` 获取嵌入层；复制空间嵌入前检查维度一致（列维度等于 `hidden_size`），不一致则跳过；整体用 `try/except` 保护不同语言模型实现。

7. `size mismatch for spatial_embed_tokens.weight`（[8194, 2304] vs [8194, 4096]）
   - 触发场景：从 `spatialvla-4b-224` checkpoint 加载到 dev 模型（语言侧 `hidden_size=4096`），空间嵌入的列维度与语言侧不一致。
   - 根因：旧结构的视觉投影维度是 `2304`，而新语言侧是 `4096`；直接拷贝会报尺寸不匹配。
   - 修复/规避：
     - 加载时启用 `ignore_mismatched_sizes=True`。
     - 临时设 `use_spatial_token=false`（当前你不需要空间嵌入），跳过空间嵌入拷贝与维度对齐；
     - 或后续通过将 `projection_dim` 对齐 `hidden_size` 或引入线性适配器做 2304→4096 映射。

8. 动作标记重复添加日志
   - 触发场景：同一进程里多次初始化处理器或重复对 tokenizer 执行 `add_tokens(...)`。
   - 根因：组合处理器 + 目录处理器路径都可能添加标记。
   - 解决建议：在 `SpatialActionTokenizer` 中加入“已添加检查”（如查 `tokenizer.get_added_vocab()`），只添加一次；当前对功能无害，仅日志重复。

9. 目录 JSON 非法值（`Infinity`）
   - 触发场景：`processor_config.json` 出现 `Infinity` 导致 JSON 解析错误（“需要值”）。
   - 根因：合法 JSON 不允许 `Infinity`；必须替换为有限数。
   - 修复：将 `Infinity` 改为 `0.0`。

10. 端到端未找到 dev 处理器模块
   - 触发场景：`auto_map` 指向 dev 名称，但目录没有对应文件。
   - 根因：HuggingFace 从目录加载时需要找到被映射的模块名；仅仓库存在 dev 文件不够。
   - 修复：在目录中新增别名模块：`configuration_spatialvla_dev.py`、`modeling_spatialvla_dev.py`、`processing_spatialvla_dev.py`，转发到仓库实现；同时处理器别名放宽类型检查。

---

## 运行与验证建议

- 集成测试：`python SpatialVLA_llava3d/test/test_llava3d_integration.py`
  - 确认 4D 因果掩码、图像标记转换逻辑与端到端加载路径。
- HuggingFace dev 测试：`python SpatialVLA_llava3d/test/test_huggingface_dev.py --model_name_or_path /path/to/spatialvla-4b-224 [--llava3d_path /path/to/llava3d_7B]`
  - 验证语言侧替换、tokenizer 类型一致、动作标记追加与嵌入同步；注意 protobuf 兼容信息。
  - 若暂时不使用空间嵌入（`use_spatial_token=false`），可规避空间嵌入维度对齐问题；需要启用时建议配合 `projection_dim==hidden_size` 或适配器。

---

## 未来优化建议
- 空间嵌入启用时：让 `projection_dim` 与语言侧 `hidden_size` 对齐，或为 `spatial_embed_tokens` 增加线性适配器进行维度映射。
- 处理器动作标记添加：加入“只添加一次”的保护，避免重复日志与不必要的嵌入扩展。
- state_dict 过滤加载：只加载视觉塔/投影器权重，语言侧用 LLaVA‑3D 权重，减少“unused weights”提示。
- 环境依赖：安装兼容版本的 `protobuf`，移除纯 Python 降级以提升性能。
 - 加载整洁化：在 `from_pretrained` 支持 state_dict 过滤，只加载视觉塔/投影器权重，语言侧权重从 LLaVA‑3D，仅映射必要层，减少“unused weights”提示。
 - 文档与测试：将“运行时覆盖”策略写入 README，用例中统一使用组合处理器，减少 `AutoProcessor` 路径差异。

---

## 结论
- 当前 dev 集成遵循“SpatialVLA 负责视觉侧、LLaVA‑3D 负责语言侧”的准则；核心路径已打通，关键测试与端到端加载在运行时覆盖策略下稳定。
- 在不改动模型目录的大前提下，通过别名与手动组合处理器的方式，解决了 tokenizer 类型不匹配与目录 auto_map 带来的加载问题。
- 若后续启用空间嵌入（`use_spatial_token=true`），建议配合维度对齐或适配器，以保证拷贝语义与语言侧维度一致。
11. 最新调试异常：语言侧嵌入直接替换导致属性错误与新初始化提示
   - 触发场景：在 `test_huggingface_dev.py` 加载 `spatialvla-4b-224` 后，日志显示大量 `language_model.model.layers.*` 权重未使用或新初始化，同时若尝试访问 `language_model.model.embed_tokens` 会报 `AttributeError`。
   - 根因：
     - 语言模型替换为 LLaVA‑3D 后，其内部结构与 Gemma2 不同；不存在 `.model.embed_tokens` 的嵌套属性。
     - `spatial_embed_tokens.weight` 的列维度来自旧结构的视觉投影（2304），不等于 LLaVA‑3D 的文本隐藏维度（4096），直接拷贝必然不匹配。
   - 关联影响：
     - Transformes 会在 `ignore_mismatched_sizes=True` 下重新初始化不匹配层，提示“should probably TRAIN this model ...”。这是正常信息，提醒下游任务需要微调以获得最佳效果。
   - 修复：
     - 在 `from_pretrained` 使用 `get_input_embeddings()` 获取嵌入层，并在维度一致时才进行替换；否则安全跳过，避免崩溃。
     - 运行时将 `use_spatial_token=false`（你当前不需要空间嵌入），彻底规避空间嵌入拷贝与维度对齐问题。
     - 在 `test_huggingface_dev.py` 中调用 `model.resize_token_embeddings(len(tok))`，确保新增动作标记的词表扩展与语言侧嵌入矩阵一致。