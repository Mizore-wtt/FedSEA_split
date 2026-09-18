# 缓存与会话接口

[返回项目入口](../README.md) | [第三阶段操作](../3_kv_cache/README.md)

## 先理解变化

KV Cache 是每一层保存的注意力历史，不是模型权重，也不是把之前的回答直接复制出来。
以输入 100 个 token、生成 4 个 token 为例：

```text
第二阶段：每次进入 A/B/C 的长度 = 100, 101, 102, 103
第三阶段：每次进入 A/B/C 的长度 = 100,   1,   1,   1
```

第一步叫 prefill，处理完整新输入；后面叫 decode，只处理新 token，
并读取缓存里的历史 K/V。模型需要额外内存保存缓存；切分本身不会减少总权重大小。

## 三段归属

`PartitionedCache` 是兼容 Transformers 的总视图，内含独立的 A/B/C 缓存对象。
每个对象只保存自己所属层的 K/V，底层仍用 Transformers 4.56.1 的
`DynamicCache` 所创建的原生 `DynamicLayer`，不重写注意力或 K/V 拼接算法。
总视图引用这些相同对象，不是第四份缓存。

原模型层号保持不变。例如 p=2 时，中段第一层的原始索引仍为 2（人类层号 3）；
只有中段缓存入口把它映射到自己的第 0 个槽位，越界写入直接拒绝。
前向使用模型原有 Decoder、RoPE、因果掩码和 `generate()` 贪心算法。

每个 split 模型实例有自己的缓存身份标记，不能拿另一个模型实例的缓存混用。
同一个模型可以创建多个独立会话，但本地交互入口一次只管理一个会话。
第三阶段不是线程安全服务。第四阶段已独立实现服务端会话表、计算锁和网络序列化，
不改变本页描述的第三阶段本地缓存 API，详见 [HTTPS 接口](https.md)。

## 开发入口

| 接口 | 作用 |
|---|---|
| `build_cached_model(reference, partition, adapter)` | 用原有只读权重创建缓存版三段模型 |
| `split.new_cache()` | 创建本模型的空 A/B/C 缓存 |
| `CachedSession(split)` | 创建独立会话，管理 token 与缓存的一致性 |
| `session.generate(inputs, generation_config)` | 接收完整当前提示词，自动处理可复用前缀 |
| `session.reset()` | 清空三段缓存及已缓存 token/mask |
| `session.cache.describe()` | 查看每段层范围、token 数和实际 K/V 字节数 |

会话的 `inputs` 为单会话的 `input_ids` 与 `attention_mask`，
均为完整提示词形状 `[1, sequence]`，不是只传最后一个 token。
会话层负责将历史缓存交给 Transformers，由其只计算尚未缓存的后缀。
底层模型 `forward` 则接收尚未处理的新 token 和覆盖历史+新增位置的完整 mask。

## 多轮、停止与重置

下一轮先重新套用模型自己的聊天模板，再逐项核对新提示词和已缓存历史的
token ID、mask。相同前缀复用；末尾因模板或分词变化而不同则裁剪到共同前缀，
完全不同就重建。不能只用“文字看起来一样”判断缓存可复用。

`generate()` 选出的最后一个 token 通常尚未再次送入模型，因此返回时
`cached_tokens = 输出完整序列长度 - 1`。即使最后一个 token 是 EOS 也如此。
下一轮会把尚未处理的 token 与新内容一起补算，不能人为把缓存长度加一。
重复提交同一提示词时，至少保留最后一个提示 token 重算，以取得下一 token logits。

`/reset` 丢弃对话与缓存；退出释放会话；生成异常也清除缓存。
若某段已经写入但后段失败，半完成缓存标记为无效，必须重置后再用。
输入超长会报错，不自动截断或静默删除上下文。

## 报告字段

| 字段 | 含义 |
|---|---|
| `reused_prefix_tokens` | 本轮从旧会话复用多少提示 token |
| `prefill_tokens` | 当前提示词中还需要送入模型计算的 token 数 |
| `query_tokens_per_forward` | 每次模型前向的输入长度；正常为 `[prefill, 1, 1, ...]` |
| `cached_tokens` | 三段已实际计算并缓存的序列长度 |
| `final_kv_cache.layout` | 每段层范围、层数、缓存 token 数及 `kv_bytes` |
| `forward_checks.steps` | prefill 与后续增量步的边界、logits、每层 K/V 对比 |
| `multi_turn` | 连续多轮复用及重置结果 |

`kv_bytes` 只计算实际 K/V 张量，不是总显存；不包括权重、临时激活、
CUDA 分配器保留内存或 Python 对象。对比还采集 logits、复制缓存做数值检查，
耗时不等于正常推理吞吐。第三阶段比较当前完整缓存模型，不读取第一阶段历史数组。

## 当前支持范围

已提供 Qwen2 与 Qwen3-MoE 的完整注意力适配；固定 eager attention、
动态非量化缓存、贪心生成。聊天为 batch=1；底层前向另有左侧 padding 的小模型测试。
不支持滑动/混合注意力缓存、静态缓存、缓存卸载、缓存量化、beam search、
采样、辅助解码、完整 attention/router 输出或持久化缓存。
这些路径会尽量提前拒绝，后续扩展需要单独实现和测试。

真实验证使用本地 Qwen2.5-0.5B；Qwen3-MoE 验证使用随机小模型，
不是实际 30B 部署。HTTPS 已在第四阶段实现；噪声和多设备放置仍未实现。
