# 第五阶段验证记录

[启动说明](README.md) | [指标口径](../docs/noise.md)

实施及验收日期：2026-09-08。正式项目：`E:\0_wt_ws\FedSEA`。
本页记录实际运行结果，不将“实验完成”解释为噪声没有质量影响或已获得隐私保证。

## 环境和范围

真实权重为现有 Qwen2.5-0.5B-Instruct，未重新下载。
使用 RTX A3000 Laptop GPU、CUDA/FP16、eager attention、Transformers 4.56.1、
PyTorch 2.8.0+cu126，推理生成上限默认 128 token。
沿用现有本地 CA 和令牌，所有实际推理通过两个独立进程之间的回环 HTTPS 完成。

Qwen3-MoE 仅做随机小模型测试，真实 Qwen3-30B 权重仍未下载或部署。
没有训练、联邦聚合、模型反演攻击或差分隐私预算评测。

## 自动测试

正式目录执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 .\scripts\run_tests.py --report
```

共用接口 28、第一阶段 10、第二阶段 54、第三阶段 33、第四阶段 37、第五阶段 45，
**合计 207 项全部通过，无跳过**。报告与分组日志位于
[本次测试目录](../tests/runs/20260908T063918Z_c8cda1bd/)。

第五阶段覆盖：

- Gaussian 均值/标准差、种子独立与复现、全局 RNG 不被改动、零噪声不抽样且原样返回。
- FP16/BF16/FP32、实际舍入后噪声统计、非有限/非法配置拒绝。
- Qwen2 与 Qwen3-MoE 随机模型的多切分、零噪声完整模型对齐。
- 仅上传 A 的扰动结果，不额外上传原文、token ID、种子或干净 A 输出。
- 真实 TLS 正/零噪声结果、缓存重建、不重试丢失响应、失败后的会话和 hook 清理。
- GPU/FP16 下 1、128、1024 token 长度的噪声采样和 CPU 私有随机流一致性。
- 评测覆盖、数值失败/中断落盘、低质量结果不伪装成执行错误、JSON/CSV/Markdown 输出。
- 聊天重置且不落盘、启动器拒绝错误服务实例并仅清理自己的子进程。

## 默认真实评测

执行原样默认命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate
```

实际切分 `2/20/2`，6 个问题，sigma 为 `0 / 0.02 / 0.05 / 0.1`，
基础种子 `42 / 43`，共 48 次 HTTPS 试验。
12 次零噪声试验全部与完整模型的生成 token、文字和逐步 logits 完全一致。
最终 `status: completed`，`zero_noise_gate.passed: true`。

[完整报告](runs/20260908T064344Z_7f8e022a/run.json) |
[CSV 汇总](runs/20260908T064344Z_7f8e022a/summary.csv) |
[简短汇总](runs/20260908T064344Z_7f8e022a/summary.md) |
[启动器记录](runs/launcher_20260908T064332Z_96a62465/launcher.json)

| sigma | HTTPS 试验数 | 带答案小题正确率 | 参考 token 完全一致率 | 首步 logits 平均 RMSE | 实际噪声平均 RMS |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 100% | 100% | 0 | 0 |
| 0.02 | 12 | 100% | 83.33% | 0.1802 | 0.020033 |
| 0.05 | 12 | 100% | 83.33% | 0.4127 | 0.050086 |
| 0.1 | 12 | 100% | 75% | 0.8809 | 0.100174 |

**正确率只来自 5 个很简单的带答案题，每题两个种子，共 10 个评分 trial。**
开放解释题不计入正确率。本轮非零噪声改变了全部题目的首步分数和部分回答，
但这几个短题的严格答案评分仍为 1；不能外推为通用任务/大噪声下性能不下降。
参考 token 不同也可能只是大小写等变化，因此它与规范化后的任务得分并不冲突。
以上不构成隐私或速度结论，耗时和内存的测量范围见 `docs/noise.md`。

## 切分与种子复现

将切分改为 `1/20/3`，其余保留相同模型和推理配置：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate --p 1 --q 3 --sigmas 0 0.05 --seeds 42
```

12 次试验正常完成，6 次零噪声逐步对齐；非零条件正常记录影响。
原样重复此命令后，逐项核对 12 个 trial 的问题、sigma、种子、回答、生成 token、
比较指标及实际噪声统计，全部一致；耗时和内存不作为可复现相等条件。

- [第一次报告](runs/20260908T064710Z_6c0d7c79/run.json)及[启动器记录](runs/launcher_20260908T064658Z_06a515a2/launcher.json)。
- [重复报告](runs/20260908T065114Z_a5a1a8df/run.json)及[启动器记录](runs/launcher_20260908T065103Z_150fa7d5/launcher.json)。

## 问答入口和清理

中文单问题使用 `run_local.py --prompt '请用两句话介绍联邦学习。' --sigma 0.02`，
生成了非空中文回答并保存[单次报告](runs/20260908T070145Z_ec607dbe/run.json)。
该模式只加载正常的 A/C 与 B，不额外加载完整参考模型；
它是入口验收，不是中文质量评分，见[启动器记录](runs/launcher_20260908T070134Z_db8704e0/launcher.json)。

从项目外目录直接调用 Python 一键聊天入口，使用 `3/20/1`、sigma=0.02、
输出上限 16 token，以两条固定测试输入检查问答、`/reset`、`/exit`，
分别获得 `OK` 和 `42`。该试验没有生成会话 `run.json` 或聊天日志，
只有[无对话的启动器记录](runs/launcher_20260908T065848Z_2fefb5b6/launcher.json)。
PowerShell 包装入口还单独通过了启动/EOF 退出检查；
管道批量输入测试直接传给 Python，不通过不转交管道输入的 `.ps1` 包装脚本。

以上启动器记录均为 `status: completed`、`remaining_sessions: 0`、
`server_stopped: true`。未留下需要用户手动关闭的测试服务。

## 文件保护

本次只新增第五阶段及文档，并对第四阶段共用选择器、启动提示增加兼容的扩展参数。
第四阶段的默认计算与无噪声行为保持不变，全部旧阶段测试通过。
检测到第一阶段 `baseline.py` 新增了用户注释，部署时明确跳过该文件并保留改动。
修改前源码备份位于 `E:\0_wt_ws\pre\tmp\fedsea_stage5_backup_20260908_143744_515`；
没有复制模型权重、环境、私钥，也未删除或覆盖已有实验结果。
