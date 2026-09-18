# 模型配置与扩展

[返回项目入口](../README.md)

模型选择、推理参数和切分方式分开管理。换模型不用复制一套阶段代码，
但**新架构需要适配器，新权重需要独立验证**，不是任意模型改个名字就能切分。

## 当前登记的模型

| `--model` 名称 | 官方仓库 | 架构 / 层数 | 当前状态 |
|---|---|---|---|
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | qwen2 / 24 | 已部署，默认 |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B` | qwen3_moe / 48 | 预留，未启用 |
| `qwen3-30b-a3b-instruct-2507` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | qwen3_moe / 48 | 预留，未启用 |

你提到的“Qwen3-30B”先预留了两个明确版本，后续再选择。
它们是 MoE 模型，不能当成 Qwen2.5 的普通稠密层直接套用。
原版模板预设 `enable_thinking: false`；Instruct-2507 只提供非思考模式，
不需要该开关。此项目的贪心生成用于拆分对齐，不是官方质量评测的采样配方。

## 模型清单怎么用

`configs/model.json` 最外层包含 `default_model` 和 `models`。
每个模型单独一项：

| 字段 | 用途 |
|---|---|
| `enabled` | 是否允许准备/加载；`false` 仍可预览 |
| `repo_id` / `revision` | 上游仓库和固定的 40 位 commit，不能用浮动 `main` |
| `local_dir` | 本项目 `models/` 下该模型的独立目录 |
| `model_type` / `adapter` | 实际架构与拆分适配器键；当前要求一致 |
| `num_hidden_layers` / `hidden_size` | 分层和接口预览参数；加载时对照实际 checkpoint |
| `parameter_count` | 总参数量，用于估算权重内存，不是 MoE 激活参数量 |
| `weights_sha256` | 可选，当前小模型单文件权重的固定校验值 |
| `chat_template_kwargs` | 模型特有的聊天模板开关，不与其他模型混用 |
| `baseline_run` | 可选的第一阶段历史参考；默认 `null`，不是运行模型或主对比的前提 |

模型选择优先级：**命令行 `--model` > 阶段 `config.json` 的 `model` >
全局 `default_model`**。阶段里写 `null` 就继承默认值。
不同模型目录必须不同；旧模型和旧实验结果不会自动删除。
`baseline_run` 不会自动跟随最近一次实验变化；只有需要固定历史复核时才填写。
自动参考丢失会提示跳过，显式参考路径错误则提前报错，详见[历史基线规则](metrics.md#历史基线)。

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --list-models
# 仅看模型身份，不加载
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --describe --model qwen3-30b-a3b
# 仅看大模型如何切分，结果是 2 + 44 + 2，不代表已部署
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --describe --model qwen3-30b-a3b --p 2 --q 2
```

## 推理参数放哪里

`configs/inference.json` 保存共同默认值。阶段配置的 `runtime_overrides`
可以只覆盖少数项，例如：

```json
{
  "model": "qwen2.5-0.5b-instruct",
  "runtime_config": "configs/inference.json",
  "runtime_overrides": {"max_new_tokens": 256}
}
```

上面是第一阶段完整配置示例；第二、三阶段只修改这些字段，保留它们自己的
`split`、`comparison`、`length_checks` 等字段。
推理优先级：**命令行 `--device` / `--max-new-tokens` >
阶段覆盖 > 共用默认值**。覆盖在内存中完成，不会重写其他阶段配置。
精度可选 `float16` / `bfloat16` / `float32`；CPU 当前固定 FP32。

## Qwen3-30B 启用前

两种预留模型总参数约 30.5B，每次激活约 3.3B，但本加载器仍需存放全部权重。
估算 FP16/BF16 权重约 `30.5e9 * 2 = 61 GB`（约 56.8 GiB），
还没计入缓存、激活、加载过程和运行时开销。CPU/FP32 仅权重约 122 GB。
当前 6GB 显卡及约 64GB 系统内存不适合本项目目前的单设备加载方式。

第 1–3 阶段只有单 CPU / 单 GPU 的完整加载，第四阶段已有按所属段加载，
但两个进程仍使用同一台电脑，不会减少两端合计的总权重。
尚未实现量化、CPU 卸载、多 GPU 调度或大模型部署。完整模型 GPU 加载前会估算并拒绝明显不足的显存，
但通过预检查不保证长输入也不会 OOM。

后续启用顺序：

1. 确认具体官方版本和硬件/放置方案；必要时先实现卸载或多设备支持。
2. 核实并固定 revision，检查结构参数、模板设置、推理精度，再启用该模型项。
3. 用准备脚本导入/下载完整权重及分片索引，不复用小模型目录。
4. 先运行该模型的完整基线并保存数组，将 `baseline_run` 指向新记录。
5. 再按多种 p/q 运行拆分对比；真实大权重通过前不能称为部署完成。

## 开发接口

`fedsea/model_registry.py` 管身份与启用校验，`fedsea/runtime.py` 管统一本地加载
和分词，`fedsea/partition.py` 管与总层数相关的切分。
`2_model_sep/split_model.py` 的 `SPLIT_ADAPTERS` 注册具体架构包装器：
当前有 `qwen2`、`qwen3_moe`，后者保留原有专家、路由和 Transformer 层对象。
两者都不复制权重，不支持分段 KV Cache 或路由输出追踪。

第三阶段的缓存适配器独立位于 `3_kv_cache/cached_model.py`，
通过 `CACHED_ADAPTERS` 注册 `qwen2` / `qwen3_moe`，复用相同模型与切分清单。
缓存归属和会话生命周期见[缓存接口](cache.md)；第二阶段继续保持无缓存对照，
不能向其包装器传入第三阶段缓存。

第四阶段在 `4_https_split/https_core/weights.py` 通过 `ADAPTERS` 注册
原生层、归一化和 RoPE，在 `engine.py` 通过 `REMOTE_ADAPTERS` 注册生成包装器。
客户端只加载 A/C、Embedding、Norm、LM Head；服务端只加载 B。
单文件/分片 safetensors 与 tied/untied head 均有小模型测试；
真实 30B 仍然禁用。详情见 [HTTPS 接口](https.md)。

第五阶段复用第四阶段的角色加载器和模型适配器，不新增或复制大模型权重。
`5_noise_eval/noise.py::NOISE_POLICIES` 是独立的扰动策略接口，不依赖固定隐藏维度或层数。
`--model`、`--p/--q` 保持原有含义；一键入口同时给两端相同模型/切分参数。
噪声定义和缓存隔离见[噪声实验说明](noise.md)。

同架构新增模型仍应检查模板、共享权重、位置和掩码行为；新架构需新增
`build_split_model` 能选择的包装器，并用随机小模型测试完整/拆分前向、
多步生成、padding、权重共享和多种切分，最后验证真实 checkpoint。
本次 MoE 只做随机小模型测试，没有运行真实 30B 权重。

官方元数据核对日期：2026-09-07。Hugging Face 访问超时后，
从 Qwen 的官方 ModelScope 仓库核实了 Instruct-2507 的配置和模型卡。
这里只读了元数据，未下载权重；预留 revision 仍为 `null`，启用前需再固定版本。
参考位置：

```text
https://huggingface.co/Qwen/Qwen3-30B-A3B
https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/README.md
https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507/resolve/master/config.json
https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507/resolve/master/README.md
```
