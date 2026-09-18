# 代码阅读顺序

[项目入口](../README.md) | [切分配置](splits.md) | [缓存接口](cache.md) | [HTTPS 接口](https.md)

先看流程和核心函数，不必一开始就读模型库里的每个注意力公式。
下文中的 A/B/C 分别对应前 p 层、中 k 层、后 q 层。
代码保留原有注释语言：已有中文文件补中文说明，其余文件用简短英文标注关键约定。

## 先分清四类数据

| 名称 | 含义 | 典型形状 |
|---|---|---|
| `messages` | 用户、助手、系统的文字对话 | Python 列表 |
| `input_ids` | 分词器把文字编码成的整数编号 | `[batch, tokens]` |
| `hidden` / `hidden_states` | 模型内部向量，不是 token 编号或最终回答 | `[batch, tokens, hidden_size]` |
| `logits` | 每个候选下一个 token 的未归一化分数 | `[batch, tokens, vocab_size]` |

当前聊天 `batch=1`；小模型 `hidden_size=896`。
KV Cache 保存各层已经计算过的注意力 K/V，不是权重，也不是直接保存最终回答。
第一次处理新输入叫 prefill；之后逐步生成叫 decode。

## 共用配置和模型加载

1. `fedsea/project.py::load_runtime_config`：读取共用配置，再合并阶段覆盖，不修改原 JSON。
2. `fedsea/model_registry.py::ModelRegistry.get`：选择模型身份和模板；禁用的 30B 只允许预览。
3. `fedsea/partition.py::resolve_partition`：先确定 p/q，再按总层数计算 k。
4. `fedsea/runtime.py::ModelRuntime`：第 1–3 阶段共用的完整模型加载器。
5. `ModelRuntime.tokenize`：按模型自己的聊天模板把 `messages` 变成 token 和 mask。

各路径都以项目根目录为基准，不是以打开终端的位置为基准。
`.eval()` 关闭训练行为，`torch.inference_mode()` 避免建立梯度图；这里没有训练步骤。

## 第一阶段：完整模型

入口是 `1_model_base/baseline.py::main`，日常聊天走 `chat -> Baseline.generate`。
`model.generate()` 内部重复计算下一 token，返回“输入 token + 新生成 token”；
代码需要截去输入部分，才能解码得到本轮回答。

本阶段每轮重新处理完整对话，只在同一轮生成过程中使用普通缓存。
不要把它与第三阶段的跨轮缓存复用混淆。

## 第二阶段：单程序三段

入口是 `2_model_sep/run_split.py::main`：

```text
读取配置 -> 可选历史记录预检查 -> 加载一次完整模型
                                  |
                          build_split_model
                                  |
       run_comparison：完整模型执行一次，再让 A/B/C 执行一次
```

在 `split_model.py` 中按这个顺序阅读：

1. `initialize_split`：只初始化包装器，A/B/C 引用原模型的层对象，不复制另一套权重。
2. `FrontStage.forward`：Embedding 把整数 token 变成向量，准备位置/掩码，运行 A。
3. `BlockStage.forward`：依次调用本段的原生 Transformer 层。
4. `BackStage.forward`：运行 C，再做最终归一化；原生 LM 包装器随后用 `lm_head` 产生 logits。
5. `SplitBackbone.forward`：连接 A → B → C，三段始终用同一份位置与掩码上下文。

这里关闭缓存，所以每生成一个 token 都会重新计算整个已有序列。
B 不能把位置从 0 重新编号，否则拆分后的数学计算就变了。

`comparison.py` 中的 `check_prefill` 用临时 hook 记录 A/B/C 边界；
`check_generation` 比较逐 token 分数和生成序列。仅回答文字一样不够，数值也要检查。
`history.py::load_reference` 则处理**可选**的第一阶段旧记录，和本轮主对比分开。
历史 `loaded` 只是成功读入，`skipped` 是未比较，都不能当作历史一致性通过。

## 第三阶段：缓存归属

看 `3_kv_cache/cached_model.py::CachedBackbone.forward` 和 `partition_cache.py`：

- `cache.a / cache.b / cache.c` 分别接收自己负责的层写入的 K/V。
- 层号仍是原始全局索引，只在缓存入口转换成本段槽位。
- 模型前向的 `hidden` 只包含新 token，mask 却必须覆盖“历史 + 新 token”。
- 总缓存视图引用同一批原生缓存对象，不是额外复制一份所有 K/V。

再看 `session.py::CachedSession`：新一轮重新分词，对比旧 token 和 mask 的精确前缀。
前缀相同就复用；后缀不同就裁剪；完全不同就重建。
最终生成的 token 尚未再次进入模型，因此缓存长度通常等于输出序列长度减一。
不要因为回答已包含最后一个字就人为把缓存长度加一。

## 第四阶段：两个程序

启动入口分别是 `4_https_split/server.py` 和 `client.py`。
正常客户端不构造完整参考模型，只有 `--compare` 才额外加载完整模型做对照。

| 文件 / 接口 | 重点 |
|---|---|
| `https_core/weights.py::RoleWeights` | meta 只搭结构，随后只读取所属层参数；保持共享 Embedding/Head 和原生 RoPE 初始化路径 |
| `engine.py::RemoteBackbone.forward` | 本地算 A，调用 HTTPS 的 B，再继续算 C |
| `engine.py::ServerState.dispatch` | 服务端操作会话表、B 层及 B 缓存；锁保护计算和缓存变更 |
| `engine.py::RemoteCache.operation` | 同时核对会话、请求序号和缓存长度，确认回包后再更新本地状态 |
| `protocol.py::pack / unpack` | JSON 元数据与 safetensors 字节；先校验消息/维度预算再反序列化 |
| `transport.py::HttpsRpc.call` | 校验证书、发送令牌、执行一次请求；不自动重试会改变 B 缓存的前向 |
| `session.py::HttpsSession` | 本地保存对话 token，同步两端缓存裁剪、重置和失败清理 |

服务端“可能已经计算成功，只是响应丢失”时，不能直接再算同一请求，
否则 B 的缓存可能比 A/C 多一段。此时清空本地状态，尽力关闭服务端会话；
不可达的残留由 TTL 到期回收。

## 第五阶段：上传前加噪

先读 `5_noise_eval/run_local.py`：它自动启动第四阶段的 B 服务，再启动第五阶段客户端，
并在结束或异常时关闭自己创建的进程。`run_noise.py` 则只连接一个已经运行的服务。

加噪计算看 `noise.py::GaussianNoise.apply`：对 A 输出做 FP32 高斯扰动后转回模型精度；
零噪声直接返回原张量。`describe()` 统计的是实际传输数值与原值的差，不是原始随机样本。
每个会话使用独立随机数生成器，基础种子与问题 ID 共同决定实际种子。

接入位置看 `noisy_session.py::NoiseSession.reset`：
它只给本会话缓存的 RPC 套上 `NoisyRpc`，不改共享模型或第四阶段序列化器。
上传前改变 `hidden`；服务器算完 B 返回后，仍由原来代码算 C。
本阶段每个问题/每轮聊天新建缓存，回答内部仍每次只算新的 token。

再看 `evaluation.py::evaluate`、`metrics.py` 和 `reporting.py`：
完整模型参考 -> 零噪声关卡 -> 多个强度/种子 -> JSON 与汇总表。
“和原回答一样”是稳定性，“和标准答案匹配”才是此处定义的题目得分，
两者都不证明隐私安全。公式与测量限制见[噪声实验说明](noise.md)。

## 看懂测试输出

`--compare` 的 `PASS` 是本轮一致性检查通过，不是模型所有知识正确。
`stage1=skipped` 可以与主对比 `PASS` 同时出现，表示只完成本轮完整/拆分对比。
各阶段 `runs/` 是可清理的实验输出，不是模型依赖文件；请保留要复核的记录。
模型在 `models/`，环境在 `.venv/`，HTTPS 凭据在 `certs/`，不要把这些当作旧结果删掉。

执行全部自动测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --report
```

六套测试各用独立进程，避免不同阶段同名模块互相覆盖。
测试不下载真实大模型；第四、五阶段会用临时证书和回环连接，结束时清理。
