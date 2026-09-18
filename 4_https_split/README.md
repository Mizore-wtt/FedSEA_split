# 第四阶段：HTTPS 拆分推理

[返回项目入口](../README.md)

**本阶段：同机两个进程，HTTPS 通信，分段 KV Cache，暂不加噪。**
实际测试范围与结果见 [VERIFICATION.md](VERIFICATION.md)。

## 这一步做什么

```text
PowerShell 窗口 1：服务端，加载并计算中间 B 段
PowerShell 窗口 2：客户端，输入问题、计算 A/C、显示回答

客户端：文字 -> 分词 -> Embedding -> A：前 p 层
                                    |
                        HTTPS：隐藏张量、mask、位置信息
                                    v
服务端：                         B：中间 k 层
                                    |
                              HTTPS：隐藏张量
                                    v
客户端：C：后 q 层 -> Norm -> LM Head -> 选出下一个 token -> 文字回答
```

生成一句话需要重复多次上述过程，不是只往返一次。首次处理新提示词，
之后通常每次处理一个新 token；A/C 缓存留在客户端，B 缓存留在服务端。
模型仍然使用共用 `models/`，但两个正常推理进程各自只读取自己负责的参数。
小模型默认 `2+20+2`。Embedding 和输出头留在客户端，不计入 p/q。

## 如何启动

以下命令均在 PowerShell 中执行，不需要激活虚拟环境。

### 1. 证书准备，只需首次执行

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\prepare_tls.py
```

如果 `certs/localhost/` 已存在且证书可用，跳过这一步。
脚本不会覆盖已有凭据，不修改 Windows 全局信任，不下载依赖。
本机使用已有 OpenSSL；其他电脑找不到时用 `--openssl '完整的openssl.exe路径'`。

### 2. 窗口 1 启动服务端，保持打开

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\server.py
```

看到 `"status": "ready"` 后再开客户端。默认地址 `https://localhost:8443`，
只监听 `127.0.0.1`，浏览器不是聊天入口，不必在浏览器打开。

### 3. 窗口 2 启动客户端

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --chat
```

输入问题后回车，等待完整回答。`/reset` 清空对话及两端三段缓存，
`/exit` 退出客户端并释放远程会话。随后在服务端窗口按 `Ctrl+C` 停止服务。
聊天不保存对话；网络异常会清缓存，已有对话文字仍只保留在客户端内存中。
修复连接后重新提交问题即可，不会自动重试可能已执行的推理请求。

也提供同目录的 `prepare_tls.ps1`、`run_server.ps1`、`run_client.ps1` 快捷脚本，
参数原样传给 Python。若系统拦截 ps1，直接使用以上 Python 命令即可。

## 对比、单次问答和换切分

服务端已启动时：

```powershell
# 与完整模型逐步对比，保存测试记录
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --compare
# 单次问答，保存问题、回答和指标
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --prompt '用一句话介绍联邦学习。'
# 只预览切分，不加载、不连接
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --describe --p 1 --q 3
```

**只有 `--compare` 会在客户端额外加载完整参考模型**，所以显存比正常聊天高。
对比的是当前完整模型，不用旧基线文件冒充本轮结果。结果位于
`4_https_split/runs/<UTC时间_随机编号>/run.json`，不会覆盖历史目录。

切换组合时，先退出原客户端、停止原服务端，再在两个窗口分别执行：

```powershell
# 窗口 1
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\server.py --p 1 --q 3
# 窗口 2
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --chat --p 1 --q 3
```

两边 `--model`、`--p`、`--q`、精度必须一致；不一致会拒绝建立会话。
也支持 `--split p1_q3`、`--list-models`、`--list-splits`。
仅预览大模型接口：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\client.py --describe --model qwen3-30b-a3b --p 2 --q 2
```

Qwen3-MoE 有小型随机权重测试，不代表已经下载、部署或验证真实 30B 模型。
长期设置仍用共用配置和本阶段 `config.json`，详见[模型接口](../docs/models.md)、
[切分接口](../docs/splits.md)。

## 文件结构

```text
4_https_split/
|-- README.md              操作总入口
|-- VERIFICATION.md        实际验证结果与未验证范围
|-- config.json            本阶段切分、HTTPS 地址、超时、容量等限制
|-- prompts.json           多轮记忆、复用、重置测试
|-- prepare_tls.py/.ps1     生成独立开发证书和随机访问令牌
|-- server.py              服务端启动入口，仅加载 B
|-- client.py              客户端启动入口，加载 A/C、分词、聊天
|-- run_server.ps1          服务端快捷入口
|-- run_client.ps1          客户端快捷入口
|-- verify.py              完整模型与 HTTPS 模型对比，仅测试时用
|-- verify_local.py        自动启动私有测试服务、双进程对比、结束后自动清理
|-- https_core/
|   |-- settings.py        统一读取模型、切分和阶段配置
|   |-- weights.py         按所属层选择性读取权重，架构适配器注册
|   |-- protocol.py        二进制消息、张量形状和大小校验
|   |-- transport.py       TLS、认证、HTTP 服务和客户端
|   |-- engine.py          A/B/C 执行、两端缓存、会话操作
|   |-- session.py         客户端多轮历史与前缀复用
|   |-- runtime.py         客户端加载、聊天模板和生成设置
|-- tests/                 小模型、真实 TLS、异常处理测试
|-- runs/                  自动生成的结果，不进入版本库
```

不复制权重到本阶段；证书和令牌放 `../certs/localhost/`。
正常推理不依赖前三阶段的实现；仅 `verify.py` 复用第三阶段已测过的完整模型会话策略。

## 配置与排错

`config.json` 的 `https` 管网络设置；默认每个消息最多 16 MiB、上下文 1152 token、
最多 4 个会话、600 秒空闲到期、60 秒网络读写超时。输入默认不超过 1024，
输出预算 128。调整输入/输出预算时，同步增大两端 HTTPS 上限，并考虑显存开销。
上限不是自动截断；当前限制 batch=1、贪心生成、eager attention 和动态缓存。

| 现象 | 处理 |
|---|---|
| 连接失败 | 确认先启动服务端、出现 ready，两边端口相同 |
| 8443 被占用 | 两边都加 `--port 8444`，不停止其他程序占用的端口 |
| 模型/切分/精度不一致 | 重启两边并使用相同选择；CPU 必须两边都加 `--device cpu` |
| 找不到证书或令牌 | 首次运行 `prepare_tls.py`，检查 `credentials_dir` |
| 证书过期或需换令牌 | 停止两端，在新的 certs 子目录生成凭据，再修改 `credentials_dir` |
| 会话过期 | `/reset` 后重试；长时间不操作会按 TTL 自动释放 B 缓存 |
| 输入超长 | 缩短输入或 `/reset`；不会静默丢弃历史 |
| CUDA 内存不足 | 先关闭其他推理进程；`--compare` 额外占用完整模型显存 |

证书轮换示例：`prepare_tls.py --directory certs/localhost-02`。
不要通过关闭证书校验来排错。不要公开私钥、令牌或包含敏感输入的 `runs/`。

## 测试与边界

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s .\4_https_split\tests -v
```

单元测试使用临时小模型和临时证书，会创建本地测试监听并在结束时关闭。
测试不下载权重、不影响日常证书、不需要先启动日常服务端。

本地证书已准备好后，也可一条命令跑真实模型验证，不必手动开启日常服务端：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\verify_local.py
.\.venv\Scripts\python.exe -X utf8 .\4_https_split\verify_local.py --p 1 --q 3
```

脚本自动选择空闲临时端口，只终止自己启动的测试进程。
`runs/launcher_*/` 保存启动日志，模型对比结果仍在对应 `runs/<时间_编号>/run.json`。

加噪实验已单独实现于[第五阶段](../5_noise_eval/README.md)，本阶段继续作为无噪声对照。
第五阶段复用本服务端，只有客户端上传的 A 输出发生变化，服务端自身不生成噪声。

这是**本机研究原型**，不是面向公网的生产服务，也尚未实现跨机器部署。
HTTPS 保护传输，服务端仍能看到明文隐藏张量和长度/位置信息，
不能据此承诺隐藏输入或差分隐私。没有加噪、训练、联邦聚合、量化、自动切分、
故障后跨进程缓存恢复或多 GPU 调度。协议与安全细节见 [docs/https.md](../docs/https.md)。
