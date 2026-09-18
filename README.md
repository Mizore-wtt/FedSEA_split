# FedSEA：本地分阶段复现实验

先跑通完整模型，再验证前、中、后三段计算，之后接入缓存、HTTPS 和噪声。
目前可用的是 **Qwen2.5-0.5B-Instruct 的完整推理、单进程拆分、三段 KV Cache、同机 HTTPS 双进程推理和 A 输出加噪评测**。
Qwen3-30B 的模型配置和 MoE 拆分接口已预留，**没有下载或部署 30B 权重**。

## 先启动一次

本机工作目录是 `E:\0_wt_ws\FedSEA`。打开 PowerShell，执行：

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
# 完整模型聊天
.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --chat
```

输入问题后回车。`/reset` 清空历史，`/exit` 退出。聊天默认不保存对话。
不必激活虚拟环境，也不必重复安装或下载模型。
换电脑或环境缺失时，先看[安装说明](docs/setup.md)。

退出聊天后，可以验证拆分：

```powershell
# 前2层 + 中间自动计算 + 后2层，与完整模型对比
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare
# 换成前1层、后3层
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --p 1 --q 3
# 使用拆分模型聊天
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --chat
```

看到各项 `PASS`，且最终 `run.json` 中 `status` 为 `passed`，表示本轮对比通过。
这表示测试中的拆分计算一致，不表示模型回答一定正确。
第二阶段默认比较**本轮加载的完整模型**，不要求旧的第一阶段 `runs/` 存在。
`stage1=skipped` 仅表示没有复核旧记录，不影响本轮主对比。

第三阶段已加入分段缓存，日常体验拆分聊天可用：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --chat
.\.venv\Scripts\python.exe -X utf8 .\3_kv_cache\run_cached.py --compare --p 1 --q 3
```

首次处理完整输入，后续每步只算新 token；`/reset` 同时清空对话与三段缓存。
操作与文件布局见[第三阶段 README](3_kv_cache/README.md)。

第四阶段把中间 B 段移入另一个进程，通过 HTTPS 往返，仍不加噪。
需要先开服务端，再开客户端，首次需准备本地证书：

```powershell
# 首次生成；certs/localhost 已有可用凭据时跳过
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\prepare_tls.py
# 窗口 1，保持运行
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\server.py
# 窗口 2
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --chat
```

详细的两窗口操作、修改切分、停止和排错见[第四阶段 README](4_https_split/README.md)。

第五阶段已接入 A 输出高斯噪声。推荐自动启动/关闭临时 HTTPS 服务的一键入口：

```powershell
# 默认 6 题、4 个噪声强度、2 个种子；先验证零噪声，再测回答和资源变化
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate
# 加噪聊天，不保存对话
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --chat --sigma 0.02
```

首次需已准备本地证书。第五阶段每轮重新建立缓存，一轮回答内部仍使用 KV Cache。
它完成的是推理实验链路，不是整篇论文的训练、隐私攻击与性能复现。
启动、结果及边界见[第五阶段 README](5_noise_eval/README.md)。

## 文件结构

```text
FedSEA/
|-- README.md                 总入口：启动、进度、目录、阅读导航
|-- configs/
|   |-- model.json            模型清单、默认模型、版本和本地权重位置
|   |-- inference.json        共用推理参数：设备、精度、输入/输出上限
|   |-- splits.json           切分预设：只写 p、q，k 自动计算
|-- fedsea/                   共用代码：模型加载、配置、切分参数、权重检查
|-- models/                   共用权重；每个模型一个子目录
|   |-- Qwen2.5-0.5B-Instruct/ 当前已部署的小模型
|-- 1_model_base/             完整模型：代码、阶段配置、测试、runs/
|-- 2_model_sep/              三段模型：代码、阶段配置、测试、runs/
|-- 3_kv_cache/               分段 KV Cache、会话复用、对比测试、runs/
|-- 4_https_split/            HTTPS 双进程：客户端 A/C、服务端 B、测试、runs/
|-- 5_noise_eval/             A 输出高斯噪声、单次/聊天/强度与种子评测、runs/
|-- docs/                     安装、模型扩展、切分、指标和验证记录
|-- tests/                    共用接口与命令行测试，不加载真实权重
|-- scripts/prepare_model.py  按模型配置导入/下载并校验权重
|-- scripts/run_tests.py      一个入口执行全部或选定阶段的自动测试
|-- requirements/             固定依赖和本机版本记录
|-- certs/                    本地 HTTPS 证书与令牌；凭据不进入版本库
|-- .venv/                    本项目 Python 环境
|-- .cache/                   安装和模型下载缓存
```

每个阶段放自己的代码、配置、测试和结果。第 1–3 阶段共用 `fedsea/` 的加载器；
第五阶段复用第四阶段的角色加载器与 HTTPS 引擎，只新增加噪和评测逻辑，不复制另一套网络实现。
所有阶段都不重复存放权重。`runs/` 为自动生成的结果，历史结果不会被本次运行覆盖。

## 想修改什么，去哪里

| 需求 | 入口 |
|---|---|
| 完整模型聊天、基线测试、结果位置 | [第一阶段 README](1_model_base/README.md) |
| 拆分聊天、完整/拆分对比 | [第二阶段 README](2_model_sep/README.md) |
| 带缓存的拆分聊天、增量生成、多轮复用 | [第三阶段 README](3_kv_cache/README.md) |
| 两个进程通过 HTTPS 配合回答 | [第四阶段 README](4_https_split/README.md) |
| 加噪聊天、噪声强度/种子评测、CSV 汇总 | [第五阶段 README](5_noise_eval/README.md) |
| sigma 定义、缓存隔离、质量/稳定性指标、隐私边界 | [噪声实验说明](docs/noise.md) |
| 第五阶段真实模型验收 | [噪声验证记录](5_noise_eval/VERIFICATION.md) |
| HTTPS 消息格式、安全边界和扩展接口 | [HTTPS 协议](docs/https.md) |
| 第四阶段真实双进程验证 | [HTTPS 验证记录](4_https_split/VERIFICATION.md) |
| 换模型、了解 Qwen3-30B 预留接口 | [模型配置与扩展](docs/models.md) |
| 从哪几个函数开始看代码、各数据代表什么 | [代码阅读顺序](docs/code_guide.md) |
| 改前后层数、添加切分组合、查看优先级 | [切分配置](docs/splits.md) |
| 重建环境、准备模型、常见启动错误 | [安装说明](docs/setup.md) |
| 理解误差、速度、显存和历史基线状态 | [指标与结果](docs/metrics.md) |
| 查看第三阶段实际测试结果 | [缓存验证记录](3_kv_cache/VERIFICATION.md) |
| 查看此前模型接口整理的测试结果 | [接口优化历史记录](docs/verification_interfaces.md) |

临时修改优先用命令行。长期修改用 `configs/`；只影响某一阶段的推理设置，
写进该阶段 `config.json` 的 `runtime_overrides`，不要修改模型目录内的上游配置。

## 检查项目

```powershell
# 自动测试，不下载真实模型；第四阶段会用临时证书测试本机 HTTPS
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --report
# 仅检查第二阶段
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --suite 2
```

`--report` 将本次测试日志统一放进 `tests/runs/`，各阶段真实模型结果仍留在各自 `runs/`。
本次排查及真实重跑情况见[项目复查记录](docs/project_review.md)。

## 当前边界

| 阶段 | 状态 | 缓存 / 网络 / 噪声 |
|---|---|---|
| 1：完整模型 | 已实现 | 普通 KV Cache 开启；无网络、无噪声 |
| 2：单进程三段 | 已实现 | 主对比双方关闭缓存；无网络、无噪声 |
| 3：分段缓存 | 已实现 | 三段各存 K/V，支持多轮复用；无网络、无噪声 |
| 4：HTTPS | 已实现 | 同机两个进程，独立加载所属层与缓存；HTTPS，无噪声 |
| 5：噪声实验 | 已实现 | A 输出高斯噪声；HTTPS；每轮新缓存、轮内 KV Cache；无形式化隐私保证 |

第 1–3 阶段只读本地模型，不下载、不监听端口。第 4 阶段也不下载，
但服务端会监听本机回环 HTTPS 端口，不开放到局域网或公网。`--prompt`、`--benchmark`、
`--compare` 会保存问题、回答及 token 数据；敏感输入请注意结果目录。
第二阶段目前用于正确性验证，不能拿它的耗时与第一阶段直接比较速度。
第三阶段对比也会采集内部数据，不以该耗时宣称加速。
第四阶段同样先验证正确性，不宣称速度提升或已经获得输入隐私保证。
第五阶段 `--evaluate`/`--prompt` 保存实验输入和输出；聊天不保存对话。
它测量噪声影响，不包含训练、联邦聚合或模型反演攻击评测。
