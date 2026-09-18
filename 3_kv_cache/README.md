# 第三阶段：分段 KV Cache

[返回项目入口](../README.md)

**已实现：同一进程中的 A/B/C 分段动态缓存。仍无 HTTPS、无噪声。**

第二阶段每生成一个新 token 都重新计算整个历史；本阶段首次处理完整输入，
之后每步只处理一个新 token，各段保存自己所属层的 K/V 历史。
多轮聊天也会复用经过核对的共同前缀。第一、二阶段代码和配置保持独立。

## 如何启动

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
# 三段缓存模型聊天，不保存对话
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --chat
# 与开启普通缓存的完整模型对比
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --compare
# 改成前1层、后3层，中间层数自动计算
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --compare --p 1 --q 3
# 同一切分也可通过预设选择
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --compare --split p1_q3
# 只对比自己的问题
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --prompt "请简单解释什么是人工智能。"
```

聊天中 `/reset` **同时清空对话与三段缓存**，`/exit` 退出并释放会话缓存。
也可以用 `.\3_kv_cache\run.ps1` 加相同参数；脚本被禁用时使用 Python 入口。
启动无需重新安装依赖或下载模型。CUDA 不可用时可明确加 `--device cpu`。

运行前预览，不加载真实权重：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --describe --p 2 --q 2
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --list-splits
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --describe --model qwen3-30b-a3b --p 1 --q 3
```

## 文件结构

```text
3_kv_cache/
|-- run_cached.py      统一命令行、聊天、比较测试与报告保存
|-- cached_model.py    缓存版 Qwen2/Qwen3-MoE 适配器与 A/B/C 层调用
|-- partition_cache.py 三段缓存归属、全局层号映射、清空与前缀裁剪
|-- session.py         会话前缀检查、多轮复用、生成和异常清理
|-- checks.py          逐步 logits、边界与每层 K/V 的比较
|-- config.json        阶段模型/推理选择、切分、阈值和检查长度
|-- prompts.json       连续多轮、记忆与重置测试
|-- run.ps1            PowerShell 启动入口
|-- tests/             随机小模型、缓存生命周期、比较器和 CLI 测试
|-- runs/              自动生成的结果，不覆盖历史
`-- VERIFICATION.md    本阶段实际验证结果和局限
```

仍使用项目 `.venv`、`models/` 和 `fedsea/` 共用接口。
不从第二阶段导入实现，不改原模型层号，也不复制另一套权重。
Qwen3-MoE 有独立适配入口，但真实 30B 权重尚未部署或验证。

## 配置约定

模型、推理默认值、切分预设继续分别来自 `configs/model.json`、
`configs/inference.json`、`configs/splits.json`。
第三阶段的 `model`、`runtime_overrides`、`split` 只影响本阶段，
命令行 `--model`、`--split`、`--p/--q` 的优先级与第二阶段一致。

`comparison` 默认仍为 `atol=1e-5, rtol=1e-5`。
`length_checks` 默认 1/128/512/1024；`decode_check_tokens` 默认 3，
表示每个长度的 prefill 后再检查三个单 token 增量步。
`prompt_files` 复用前两个阶段的问题数据；本阶段没有跨阶段导入代码。

## 怎样看结果

默认 `--compare` 检查 5 个固定问题、4 种输入长度和 3 个连续会话步骤。
主对比双方都开启缓存，并使用相同的输入分块顺序，检查：

- 回答文字、生成 token 和每一步原始 logits。
- prefill 和增量步的 A/B/C 边界及每层 K/V 数值。
- 三段缓存的层归属、序列长度，以及后续生成是否每步只计算一个 token。
- 多轮是否复用历史，重置后是否从空缓存开始。

全部通过时 `run.json` 的 `status` 为 `passed`；不一致返回非零退出码。
结果在 `runs/<UTC时间_随机标识>/`。`case_XX/` 和 `turn_XX/` 保存输入/输出
token 与首步 logits 数组；报告保存比较统计，不保存原始 K/V 张量。
问题、回答、token 和缓存都可能暴露输入信息，不能当作匿名数据。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s .\3_kv_cache\tests -v
```

缓存原理、会话接口与限制见[缓存说明](../docs/cache.md)；
实测见 [VERIFICATION.md](VERIFICATION.md)。本阶段验证正确性和增量计算，
不把带数据采集的对比耗时当作加速结论。下一步才是 HTTPS 双进程通信。
