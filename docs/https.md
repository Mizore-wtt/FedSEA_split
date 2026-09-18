# HTTPS 协议与扩展接口

[返回项目入口](../README.md) | [启动与排错](../4_https_split/README.md)

## 信任边界

本阶段仅允许 IPv4 回环监听 `127.0.0.1`，客户端只接受 `localhost` / `127.0.0.1`
的 HTTPS URL。所有端点使用随机 256-bit bearer token；TLS 最低 1.2。
客户端显式信任项目 CA，同时验证证书链、有效期和主机名。
没有关闭校验、代理、重定向、HTTP 降级或自动前向重试路径。
开发凭据目录限制为当前 Windows 用户读写，私钥和令牌被 `.gitignore` 排除。
这不防御同一用户下的恶意进程或管理员，也不替代生产环境的用户隔离。

使用 Python 标准库的有界线程 HTTP 服务，最多 8 个连接工作线程；
模型计算和缓存变更由锁串行化。网络超时是 socket 读写等待超时，
不是可以抢占 GPU 计算的总耗时截止时间。此实现不是经过安全审计的公网服务，
不防御所有本机资源耗尽攻击；迁移远程前需重新设计认证、配额、部署及威胁模型。

## 消息格式

所有业务端点使用 POST，`Content-Type: application/vnd.fedsea.safetensors`，
必须有唯一有效 `Content-Length`，不支持压缩、chunked 请求或 Expect。

```text
4 字节大端整数：JSON 元数据长度
UTF-8 JSON 对象：会话、序号、缓存长度或模型身份
可选 safetensors 二进制：至多 4 个张量
```

JSON 和 safetensors 头各限制 16 KiB，总消息受 `max_body_bytes` 限制。
拒绝重复 JSON 键、非有限 JSON 数值、异常维度、非法精度或超预算分配。
不用 pickle、`torch.load` 或任意 Python 对象反序列化。

一次 forward 只上传以下四个张量：

| 名称 | 形状 | 类型 |
|---|---|---|
| `hidden` | `[1, new_tokens, hidden_size]` | 双方约定的 FP16 / BF16 / FP32 |
| `attention_mask` | `[1, past + new_tokens]` | int64，仅 0/1 |
| `position_ids` | `[1, new_tokens]` | int64，RoPE 位置 |
| `cache_position` | `[new_tokens]` | int64，连续的新缓存槽位 |

服务端返回同精度、同形状的 `hidden`，不做有损压缩。
不上传文本、token ID、聊天历史或噪声种子。
**不上传 token ID 不等于隐藏张量不能被反推**，这是后续隐私实验要研究的内容。
隐藏张量、mask 和位置在服务端解密后可见；默认日志不打印它们。

## 会话状态机

| 端点 | 元数据 | 行为 |
|---|---|---|
| `/v1/health` | 空对象 | 已认证健康检查、进程号、模型指纹、所持有层和容量 |
| `/v1/session` | `identity` | 严格匹配模型身份，建立空 B 缓存，返回随机 session |
| `/v1/forward` | `session, seq, past` | 检查顺序、前缀和张量；计算 B 并追加缓存 |
| `/v1/crop` | `session, seq, past, keep` | 丢弃缓存后缀，保留精确公共前缀 |
| `/v1/close` | `session` | 幂等关闭，不自动创建新的远程会话 |

模型身份包括固定 revision、实测 config/权重 SHA-256、架构、p/k/q、
精度、attention 实现、Transformers 版本。启动会重新读取并检查权重校验值，
不只相信旧 manifest 中写着的指纹。该校验保证实验两边使用相同文件，
并不能证明模型文件来自可信发布者。

新会话 `seq=0, length=0`。每次成功 forward/crop，seq 加一，返回确认后的长度；
客户端再核对响应。A/B/C 使用原始全局层号，只在各段缓存入口映射到本地槽位。
两端只保存自己所持有层的 KV，不来回传输全量 KV。

本地多轮策略比较精确 token ID 和 mask 前缀；必要时同步裁剪 B 和 A/C。
HF 的最终生成 token 尚未再次进入模型，因此缓存长度通常是输出序列长度减一。
`/reset` 关闭旧 session，丢弃两端缓存；下一次推理才建立新远程会话。

错误顺序、非法 forward/crop、部分计算失败或回包失败会使所涉及会话失效。
客户端不确定服务端是否已执行时，不重放同一 forward；关闭远程会话并清空本地缓存。
如服务器暂不可达，远程残留由 TTL 回收。TTL 从最后一次成功操作算起；
过期后需用本地文字重建，不能继续复用旧缓存。健康检查不延长某个会话寿命。

## 模型与通信扩展

`https_core/weights.py::ADAPTERS` 注册原生 Decoder / Norm / RoPE 类，
`engine.py::REMOTE_ADAPTERS` 注册兼容原生 `generate()` 的客户端包装器。
当前支持 Qwen2 和 Qwen3-MoE 完整注意力。单文件、分片 safetensors、共享/不共享
embedding-head 都有随机小模型测试。原生专家、路由器和注意力计算保留不变。

角色加载器仅构造自身层的 meta 模块，然后读取所需 checkpoint 键并逐项检查形状，
不会在加载过程中先构造完整模型。RoPE 频率保持原生 FP32。
新架构、滑动/混合注意力、动态 RoPE、量化、采样等需单独适配和验证。
大模型配置仍禁用；按段加载不减少两端合计需要存放的总权重。

`HttpsRpc.call(path, metadata, tensors)` 是明确的通信边界。
第五阶段通过每个缓存自己的 `NoisyRpc` 包装器，在 A 输出上传前加噪；
通用序列化器、原模型权重及第四阶段默认会话保持不变。
详细定义和新缓存策略见[噪声实验说明](noise.md)。

## 验证与日志

正常聊天不保存；`--prompt` 和 `--compare` 会把输入/输出保存在本地 runs。
服务端无文本、张量、token、Authorization 或 session 内容日志。
健康检查也须认证，报告参数统计和层号但不返回参数或 KV 内容。
不把 MoE 成千上万个参数名称塞入健康响应，避免未来大模型超出元数据上限。

`--compare` 验证每个生成步 logits、token 序列、前缀复用计划、本地 A/C KV，
以及长输入测试中的 A/B/C 边界和 B 返回的缓存长度。
生产 API 不导出 B 缓存；随机小模型测试可在测试进程直接比较完整 A/B/C KV。
通信指标只统计应用消息体字节，不含 HTTP/TLS/TCP 额外开销。
由于启用断言、CPU 拷贝、完整模型对照和逐请求 TLS 握手，这些耗时不能当作加速结论。
