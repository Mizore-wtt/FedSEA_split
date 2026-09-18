# 指标与结果

[返回项目入口](../README.md)

本页统一说明已实现阶段的测量口径。固定的少量问题是功能检查，
不是论文数据集评测；输出非空或拆分相等都不证明回答事实正确。

## 第一阶段

| 字段 | 含义 |
|---|---|
| `model_load_seconds` | 分词器、模型加载并搬到设备的时间 |
| `first_token_seconds` | 开始生成到首个新 token 可用，包含 prefill；不含分词 |
| `generation_seconds` | 整段生成时间，含 prefill；不含加载、分词、解码、文件保存 |
| `tokens_per_second_including_prefill` | 新 token 数 / 整段生成时间，token 数包含 EOS |
| `peak_gpu_allocated_mib` | 生成期间本进程 PyTorch 峰值已分配显存，包含模型 |
| `peak_gpu_reserved_mib` | 生成期间本进程 PyTorch 峰值保留显存 |

CUDA 在计时边界同步。这里的 token/s 不是纯 decode 吞吐，显存值不是
`nvidia-smi` 的系统整体占用。`--benchmark` 先预热；单问题模式不预热。
`--save-artifacts` 会额外做无缓存前向导出首 token logits，不计入生成耗时。
比较速度时需要保持设备、精度、输入长度、生成参数、预热和采集选项一致。

## 第二阶段

主对比双方都关闭缓存，使用同一套参数与贪心解码。
检查 A 输出、B 输出、C 归一化后输出；问题 prefill 覆盖全部位置 logits，
生成期间比较每步原始 logits、token 序列和文字。
合成长度测试比较全序列边界及最后位置 logits。

`run.json` 的关键字段：

| 字段 | 怎么读 |
|---|---|
| `status` | `running` 未结束；`passed` 本轮通过；`failed` 不一致；`error` 异常；`interrupted` 用户中断 |
| `stage_config.resolved_partition` | 实际 p/k/q，不只看原始命令或预设名 |
| `model_profile` | 此次模型选择、版本和模板开关 |
| `cases[].prefill` | 输入前向与边界的检查 |
| `cases[].generation` | 每步 logits、生成 token 的检查 |
| `historical_reference` | 历史记录选择、可用性及实际读取的 run.json 哈希，独立于主对比 |
| `length_checks` | 不同合成长度的结果 |
| `exact` / `max_abs_error` | 是否逐项完全相等 / 最大绝对误差 |

默认误差阈值固定 `atol=1e-5, rtol=1e-5`，逐元素检查
`abs(expected - actual) <= atol + rtol * abs(actual)`，
与代码中 `torch.allclose(expected, actual)` 的参数顺序一致。
另外报告完全相等、平均误差和有限数值；NaN/Inf 不能通过。
跨硬件、依赖或精度不承诺逐位一致。
对比时额外采集 logits 和边界，所记耗时不是正常推理吞吐基准。

## 第三阶段

完整模型与三段模型均开启缓存，按相同的 prefill/decode 分块顺序比较，
同时检查每层 K/V、每步 logits、A/B/C 边界、多轮缓存复用和重置。
比较当前完整模型，不读取旧的第一阶段数组。
缓存长度、复用 token 数、每步输入长度和各段 K/V 字节数见[缓存说明](cache.md)。

不同计算分块可能因数值舍入而出现细小差异，因此报告明确记录双方计算顺序。
本阶段不把“重复计算整个前缀”的 logits 当作逐位相等的标准，也不放宽阈值掩盖差错。
`status: interrupted` 表示用户中断，不能视为测试通过。

## 第五阶段

`--evaluate` 先运行本轮完整模型作为参考，零噪声逐 token/logits 对齐；
非零噪声只测影响，不要求回答完全一致。每题/强度/种子使用独立的新缓存。
报告分别列出有标准答案的任务得分、参考输出一致率、首步 logits 变化、
FP16/BF16 舍入后的实际扰动、应用消息体字节、耗时和客户端显存。

`status: completed` 表示实验运行完成且零噪声关卡通过，不表示各噪声强度的回答都正确。
非零噪声回答变差是正常的待测结果；空回答也会记录，不被隐藏为程序错误。
详细公式、测量范围及隐私限制统一见[噪声实验说明](noise.md)，操作见[第五阶段 README](../5_noise_eval/README.md)。

## 历史基线

第二阶段主对比不依赖历史文件。`stage1_run: "auto"` 尝试当前模型的可选 `baseline_run`；
没有配置或该历史报告不可用时，会说明原因并标为 `skipped`，继续本轮完整模型对比。
设为 `null` 可关闭，也可使用命令行 `--stage1-run none`。
`--stage1-run '1_model_base/runs/<运行目录名>'` 或阶段配置中的具体路径属于显式选择，
若报告不存在/损坏会在加载权重前报错。两种模式都不扫描或自动改选其他“最新”记录。
路径必须是项目内 `1_model_base/runs/` 下的一次运行目录。
报告元数据的 `loaded` 只表示读入成功，每个问题仍需通过条件检查和数值对比。

| 历史状态 | 含义 |
|---|---|
| `passed` | 本问题的历史输入/输出 token、首 token logits 均通过检查 |
| `failed` | 可比较但结果不一致，会让本次对比失败 |
| `skipped` | 未配置、输入/推理/模板条件不同，或没有历史数组，未做该项检查 |
| `not_applicable` | 不是同一模型，或历史记录中没有这个问题 |

只有 `passed` 才表示与历史记录比较通过。
即使主对比通过，也要单独看历史状态，不能把跳过检查说成通过。
需要固定历史参考时，先生成该模型自己的 `--benchmark --save-artifacts` 记录，
再按需登记为 `baseline_run`；该字段为 `null` 不影响正常推理或本轮完整/拆分对比。

## 文件与隐私

每次结果写入所属阶段 `runs/<UTC时间_随机标识>/`，时间为 UTC，
与北京时间相差 8 小时；目录随机标识避免覆盖旧记录。
token 数组可以还原文字，logits 也不是匿名数据，问题、回答和环境信息都可能敏感。
聊天默认不保存会话，测试/单问题模式会落盘。数组读取用
`numpy.load(..., allow_pickle=False)`，权重、实验结果、私钥不提交版本库。
