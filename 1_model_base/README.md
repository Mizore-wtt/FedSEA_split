# 第一阶段：完整模型基线

[返回项目入口](../README.md)

目的：确认模型能够在本地正常回答，按需生成可选的历史参考结果。
后续拆分阶段会运行本轮完整模型作为对照，不要求第一阶段旧记录一直保留。
全部 Transformer 层及输入/输出模块连续运行；无拆分、无噪声、无训练。
普通 KV Cache 默认开启。

## 如何启动

不论 PowerShell 当前在哪个目录，都可以直接用完整路径：

```powershell
& 'E:\0_wt_ws\FedSEA\.venv\Scripts\python.exe' -X utf8 'E:\0_wt_ws\FedSEA\1_model_base\baseline.py' --chat
```

输入问题后回车；`/reset` 清空历史，`/exit` 退出。聊天记录默认不保存。

下面的短命令都先切换到项目根目录：

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
# 单个问题，并保存问题、回答、测量值
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --prompt "请用一句话介绍你自己。"
# 固定3个问题，预热后测试，额外导出用于对齐的数组
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --benchmark --save-artifacts
# 查看模型清单，不加载权重
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --list-models
# 启动默认聊天（Qwen2.5-0.5b）
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --chat
# 明确选择当前小模型
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --chat --model qwen2.5-0.5b-instruct
```

也可以用 `.\1_model_base\run.ps1` 加相同参数。
若系统禁止执行 `.ps1`，继续使用 Python 命令，不必放宽系统策略。

## 文件与配置

```text
1_model_base/
|-- baseline.py       完整模型生成、聊天、基线结果保存
|-- run.ps1           PowerShell 启动入口
|-- config.json       本阶段选哪个模型、继承哪些参数、局部覆盖
|-- prompts.json      固定测试问题，不是论文评测集
|-- tests/            轻量单元测试
|-- runs/             自动生成的实验记录
```

模型加载器位于 `../fedsea/runtime.py`；权重与其他阶段共用。
当前阶段配置：

```json
{
  "model": null,
  "runtime_config": "configs/inference.json",
  "runtime_overrides": {}
}
```

`model: null` 表示继承全局默认模型。默认 CUDA、FP16、eager attention、
贪心解码，输入上限 1024 token，新生成上限 128 token。
临时加 `--max-new-tokens 256` 可改生成上限；`--device cpu` 使用 CPU/FP32。
超长输入报错而不截断，CUDA 不可用也不会自动切换 CPU。
长期配置和不同模型的区别见[模型说明](../docs/models.md)。
