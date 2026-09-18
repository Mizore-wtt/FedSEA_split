# 第五阶段：隐藏输出加噪实验

[返回项目入口](../README.md)

本阶段实现 **A 输出加高斯噪声 + HTTPS 双进程推理 + 多强度/多种子评测**。
第一至第四阶段仍保持原来的无噪声对照，不复制权重，不下载新模型。
实际验收结果见 [VERIFICATION.md](VERIFICATION.md)。

## 先运行一次

本机证书已经准备好，不必重复生成。推荐一条命令自动开启临时服务、评测并关闭服务：

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate
```

默认 6 个问题、4 种强度 `0 / 0.02 / 0.05 / 0.1`、2 个种子 `42 / 43`，
共 48 次 HTTPS 试验；另有完整参考模型计算和预热。
观察 `ZERO PASS` 和 `MEASURED`，最后会打印本次 `run.json` 路径。
评测会保存问题、回答、token 和种子，不要用敏感内容做默认落盘试验。

只想先试一个问题或聊天：

```powershell
# 单问题：保存结果，不额外加载完整参考模型
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --prompt '请用两句话介绍联邦学习。' --sigma 0.02
# 聊天：不保存对话
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --chat --sigma 0.02 --noise-seed 42
# 零噪声小规模检查，再试一个非零强度
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate --sigmas 0 0.05 --seeds 42
# 改成前 1 层、中间 20 层、后 3 层；自动给服务端相同参数
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate --p 1 --q 3 --sigmas 0 0.05 --seeds 42
```

聊天支持 `/reset` 和 `/exit`。`/reset` 同时重置本实验的种子序列。
一键入口自动选择临时回环端口，退出时仅关闭自己创建的进程。
已占用的端口/已有日常服务不会被终止；显式 `--port` 被占用会报错。
也可用 `.\5_noise_eval\run.ps1` 加相同参数。

换电脑没有证书时，先按第四阶段说明执行一次 `prepare_tls.py`。
不要关闭 HTTPS 证书校验来排错。

## 数据怎么走

```text
本地文字 -> 分词 -> Embedding -> A：前 p 层
                               |
                   hidden + sigma * Gaussian(0, 1)
                               |
                           HTTPS 上传
                               v
                       服务端 B：中间 k 层
                               |
                           HTTPS 返回
                               v
                  本地 C：后 q 层 -> Norm/Head -> 下一 token
```

只对 A 的输出加噪，B 的返回不再额外加噪。不修改模型参数、token ID、mask 或位置。
首次 prefill 和后续每次 decode 都对**当前新计算的 token 隐藏向量**加噪，
不是只给整段回答加一次噪声，也不是反复给缓存中的旧向量加噪。

`sigma` 是每个元素的高斯噪声**标准差**，不是方差，也不是按隐藏向量 RMS 缩放的比例。
先以 FP32 做加法，再转回约定的 FP16/BF16/FP32 后上传。
`sigma=0` 直接返回原张量，不抽样、不做加法，不产生舍入差异。
实验会统计转回模型精度后的实际 RMS/平均绝对噪声；极小噪声可能被 FP16 舍入掉。
公式、指标口径和论文区别见[噪声实验说明](../docs/noise.md)。

## 缓存与种子

每个问题、sigma、seed 的组合都使用全新的 A/B/C 缓存和独立随机流。
一轮回答内部仍开启 KV Cache；第五阶段**不复用跨轮缓存**。
聊天会把已有文字重新分词并重新计算，避免把之前噪声条件下的隐藏缓存带进新试验。
第三、四阶段的跨轮缓存复用功能不受影响。

随机源是私有 CPU `torch.Generator`，不更改模型的全局随机状态。
根据“基础种子 + 问题 ID”派生实际种子，同一问题/种子在不同 sigma 下使用配对随机流。
同硬件、版本、输入和切分下可复跑；不同硬件/版本不承诺逐位一致。
种子只保存在本地结果，不上传服务端。它用于可复现实验，**不是密码学随机数或隐私保证**。

## 如何读结果

一次评测写入 `5_noise_eval/runs/<UTC时间_随机编号>/`：

| 文件/字段 | 含义 |
|---|---|
| `run.json` | 模型/配置/代码摘要、参考回答、每个试验的回答与完整指标 |
| `identity.p / k / q` | 本次实际切分；命令行覆盖不会反写原配置的 `split` |
| `summary.csv` | 按 sigma 汇总，Excel 可打开；对问题/种子的试验取平均 |
| `summary.md` | 简短汇总表 |
| `zero_noise_gate.passed` | 所有零噪声试验是否与本轮完整模型的 token、逐步 logits、文字一致 |
| `summary[].task_accuracy` | 有标准答案的 5 个小题，规范化后严格匹配的正确率 |
| `summary[].reference_exact_match_rate` | 与完整模型生成 token 一样的比例，不是正确率 |
| `first_token_logits` | 相同输入下的首步分数差异；后续回答分叉后不再强行逐步对齐 |
| `noise` | 实际种子、调用数、上传精度下的实际扰动幅度 |
| `communication` | 会话创建、前向和关闭的应用消息体字节，不含 TLS/TCP/HTTP 开销 |

零噪声对比失败时返回非零退出码，保存 `status: failed`，不能靠放宽误差阈值绕过。
非零噪声导致回答变化、答错或为空属于实验结果，不会伪装成程序故障。
`status: completed` 表示实验结束且零噪声关卡通过，**不是所有噪声强度都答得好**。
异常/中断分别为 `error` / `interrupted`；历史目录不会覆盖。

题目仅是 5 个带答案的英文短题和 1 个开放解释题，不是论文数据集、中文能力测试或通用质量基准。
开放题保存回答供人工查看，不自动给“语义质量分”。
耗时包含噪声统计、网络往返及评测采集；显存峰值是客户端进程的已分配量，
评测时包含额外的完整参考模型，服务端仅报告权重字节，**没有测量服务端总显存峰值**。

## 文件与配置

```text
5_noise_eval/
|-- README.md           启动、参数、结果解释
|-- config.json         模型/切分、网络、单次噪声参数、评测网格
|-- prompts.json        本阶段的固定题目与可选标准答案
|-- noise.py            GaussianNoise、独立随机源、噪声策略注册接口
|-- noisy_session.py    会话自己的上传包装器；不改共享模型或原始 A 输出
|-- settings.py         参数校验、模型/切分/网络选择
|-- execution.py        完整参考与加噪试验执行、测量范围
|-- metrics.py          数值、答案匹配、输出稳定性、分组汇总
|-- evaluation.py       零噪声关卡和多强度/种子循环
|-- reporting.py        JSON/CSV/Markdown 本地报告
|-- run_noise.py        连接已有服务：聊天、单问题、评测
|-- run_local.py        自动管理私有服务和客户端进程
|-- run.ps1             一键入口的 PowerShell 包装
|-- tests/              随机小模型、真实 TLS、指标和异常测试
|-- runs/               本阶段结果；launcher_* 为无对话的启动日志
|-- VERIFICATION.md     真实验收范围、报告位置
```

模型和推理参数继续来自 `configs/`，权重继续来自 `models/`。
本阶段 `https` 设置独立，证书默认复用 `certs/localhost`。
临时噪声：聊天/单问题用 `--sigma`、`--noise-seed`；
批量评测用 `--sigmas`、`--seeds`。评测网格必须包含 0，禁止重复项。
不要把这两类参数混用。

预览不会加载权重、读取证书或监听端口：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --describe
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --describe --model qwen3-30b-a3b --p 1 --q 3
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --suite 5 --report
```

## 两个窗口手动运行

复用第四阶段已验证的 B 服务端，并让它读取第五阶段的网络和切分配置：

```powershell
# 窗口 1，保持打开。它只算 B，不知道客户端的种子，也不会自己加噪。
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\server.py --config .\5_noise_eval\config.json
# 窗口 2
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_noise.py --chat --sigma 0.02
```

手动改模型、p/q、设备或端口时，两边都要改成相同值；一键入口会自动对齐。
`run_noise.py` 不负责启动服务，连接失败时先检查窗口 1 是否 ready。

## 当前边界

五个阶段的**推理实验链路**已实现，不等于完整复现 FedSEA-LLaMA 论文。
没有 LoRA 训练、联邦聚合、模型反演攻击、差分隐私敏感度裁剪/预算核算、
自动选择切分点、跨机器部署或生产安全审计。真实 Qwen3-30B 仍未部署。
隐藏张量经 HTTPS 解密后对服务端可见；加噪不自动证明原文无法重建。
