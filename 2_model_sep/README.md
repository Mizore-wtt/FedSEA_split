# 第二阶段：单进程三段拆分

[返回项目入口](../README.md)

目的：把同一模型分成前段 A、中段 B、后段 C，验证与完整模型计算一致。
三段仍在同一个程序、同一设备中运行，引用同一套权重。
此处不是两个进程部署，也不是把权重导出成三个文件；没有 HTTPS 或噪声。
**不需要先生成或保留第一阶段旧结果。** 默认主对比使用本轮加载的完整模型。

## 如何启动

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
# 默认前2层、后2层，执行全部一致性检查
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare
# 改成前1层、后3层
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --p 1 --q 3
# 使用同一组合的预设名，效果相同
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --split p1_q3
# 只对比你自己的一个问题，不跑合成长度测试
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --prompt "请简单解释什么是人工智能。"
# 拆分模型聊天，不保存对话
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --chat
```

聊天中 `/reset` 清空历史，`/exit` 退出。也可用 `.\2_model_sep\run.ps1`
加相同参数；脚本被禁用时用 Python 入口。

运行前预览，不加载任何模型权重：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --list-splits
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --describe --p 1 --q 3
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --describe --model qwen3-30b-a3b --p 2 --q 2
```

## 三段各算什么

层号在文档里从 1 开始。默认小模型分为：

| 段 | 内容 |
|---|---|
| A：前段，p=2 | Embedding、第 1–2 个 Transformer 层；准备位置和掩码 |
| B：中段，k=20 | 第 3–22 层 |
| C：后段，q=2 | 第 23–24 层、最终归一化、LM Head |

`k = 模型总层数 - p - q`，不需要手动维护。
Embedding、归一化、LM Head 不计入 p/k/q。共享权重关系保持不变。
A/B/C 的函数接口直接传递隐藏张量及位置、掩码信息；本阶段不使用网络序列化。
HTTPS 序列化在第四阶段单独实现。

模型和切分是两个独立选择：`--model` 决定模型，`--split` 或 `--p/--q`
决定如何分层。详见[模型接口](../docs/models.md)和[切分配置](../docs/splits.md)。

## 文件与配置

```text
2_model_sep/
|-- run_split.py       聊天、参数预览和对比入口
|-- split_model.py     A/B/C 模块、Qwen2 和 Qwen3-MoE 适配器注册
|-- comparison.py      边界、logits、生成 token 和历史基线的检查
|-- history.py         可选历史记录预检查，不把旧 runs 作为主对比依赖
|-- config.json        模型选择、切分预设、历史基线、误差阈值、测试长度
|-- prompts.json       长回答与多轮测试，补充第一阶段3个问题
|-- run.ps1            PowerShell 入口
|-- tests/             随机小模型和比较器测试
|-- runs/              自动生成的测试结果
|-- VERIFICATION.md    最初完成第二阶段时的历史记录，当前复查见 docs/project_review.md
```

共用推理配置已移到 `../configs/inference.json`，不再读取第一阶段加载器
或推理配置。`config.json` 中的 `runtime_overrides` 只影响第二阶段；
`--device`、`--max-new-tokens` 只覆盖本次执行。

## 如何判断通过

默认 `--compare` 检查 5 个问题、每步生成 logits、输出 token、回答文字，
以及 1/128/512/1024 token 合成输入的边界和最后位置 logits。
误差阈值为 `atol=1e-5, rtol=1e-5`，另记是否完全相等和最大误差。
任何主对比失败都会返回非零退出码，不应靠放宽阈值让错误“通过”。

`runs/<UTC时间_随机标识>/run.json` 保存最终 `status`、实际切分、
模型和环境、代码摘要、逐项结果、问题与回答。
`case_XX/` 保存输入/输出 token 和首 token logits 数组，注意数据隐私。
历史基线的 `skipped` / `not_applicable` 不等于“已与历史记录一致”；
具体含义见[指标与结果](../docs/metrics.md)。

## 可选的历史记录

`config.json` 的 `stage1_run` 默认是 `"auto"`，只尝试模型清单中的可选 `baseline_run`。
当前 `baseline_run` 为 `null`，默认只做本轮完整/拆分对比。
即使以后指定的自动参考目录被清理，也会明确记为 `skipped`，不会让主对比无法启动，
更不会自动猜一个“最新结果”替代它。

```powershell
# 明确不读取任何历史记录
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --stage1-run none
# 需要复核旧结果时，指定实际存在的第一阶段目录；将 <运行目录名> 换成自己的记录名
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --stage1-run '1_model_base/runs/<运行目录名>'
```

显式指定目录时若 `run.json` 缺失或格式不正确，会在加载权重前报错，不会偷偷跳过。
也可在阶段配置中写 `null` 关闭历史复核，或写项目内的具体第一阶段运行目录。
只有历史字段的 `passed` 才表示历史复核也通过，`loaded` 只是读入记录。

本阶段完整/拆分主对比**双方关闭 KV Cache**，每生成一步会重算已有序列。
第一阶段依然使用普通缓存。对比还额外采集内部数据，因此不能直接比较两阶段速度。
固定 Transformers 4.56.1，版本不匹配明确报错；更新依赖后需要重新适配验证。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s .\2_model_sep\tests -v
```

这些单元测试不加载真实 0.5B 或 30B 权重。
最新排查和重跑结果见[项目复查记录](../docs/project_review.md)，
原有[接口优化验证记录](../docs/verification_interfaces.md)保留为历史快照。
关键函数的阅读顺序见[代码导读](../docs/code_guide.md)。
第三阶段已有分段缓存、第四阶段已有 HTTPS；第二阶段仍保持无缓存对照。
