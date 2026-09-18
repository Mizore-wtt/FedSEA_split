# 切分配置

[返回项目入口](../README.md)

当前支持**连续的前、中、后三段**，三个段都至少有一层。
`p` 是前段层数，`q` 是后段层数，`k` 自动取 `总层数 - p - q`。
这里只数 Transformer 层，不数 Embedding、最终归一化或 LM Head。
不支持空段、跳层、非连续分配或自动搜索最优切分点。

本页的 `--p`、`--q`、`--split` 和优先级同时适用于第二、三、四阶段。
以下命令用第二阶段演示；启用缓存时，将入口换成
`.\3_kv_cache\run_cached.py`。第三阶段长期配置在 `3_kv_cache/config.json`，
不会改动第二阶段的切分选择。

第四阶段使用 `4_https_split/server.py` 与 `client.py`，
两边必须传相同的模型与切分参数，换组合须重启两边。
长期配置独立位于 `4_https_split/config.json`。见[双窗口示例](../4_https_split/README.md)。

## 临时指定

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
# 先预览，不加载权重
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --describe --p 2 --q 2
# 实际验证
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --p 2 --q 2
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --p 1 --q 3
# 也可以只覆盖前段，后段沿用已选配置
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --describe --p 3
```

| p / q | 24 层小模型：p+k+q | 48 层预留模型：p+k+q |
|---|---|---|
| 2 / 2 | 2+20+2 | 2+44+2 |
| 1 / 3 | 1+20+3 | 1+44+3 |
| 3 / 1 | 3+20+1 | 3+44+1 |
| 1 / 1 | 1+22+1 | 1+46+1 |
| 3 / 3 | 3+18+3 | 3+42+3 |

上表右列只是层数计算，不代表大模型已经部署。

## 保存常用组合

`configs/splits.json` 已保存 `p2_q2`、`p1_q3`、`p3_q1`、`p1_q1`、`p3_q3`。
使用预设：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --list-splits
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare --split p1_q3
```

要新增组合，只需在 `presets` 对象里加一项，例如 `"p2_q4": {"p": 2, "q": 4}`，
并保持 JSON 格式合法，不需要添加 Python 分支，也不需要写 k。

第二阶段长期默认值由 `2_model_sep/config.json` 的 `split` 指定，
替换其中这个字段即可，其余字段保留：

```json
"split": {"preset": "p1_q3"}
```

也可不用预设：

```json
"split": {"p": 1, "q": 3}
```

## 配置优先级

从低到高应用：

1. 阶段 `split` 选择预设并可覆盖其 p/q；没配置时用 `default_preset`。
2. 命令行 `--split` 会替换阶段的整个选择，包括阶段内手写的 p/q。
3. 命令行 `--p`、`--q` 最后分别覆盖，未提供的一项保持当前选择。
4. 根据选中模型的总层数计算 k，校验 p、k、q 都是正整数。

例如 `--split p1_q3 --p 2` 的最终值为 p=2、q=3。
若写出 p+q 大于等于总层数、零或负数，会在加载模型前报错。
命令行覆盖只影响本次运行，不修改 JSON 文件。

`--describe` 输出最终 p/k/q 和从 1 开始的各段层号；实际运行也会打印
`Split ready: A=..., B=..., C=...`，并把 `resolved_partition` 和
`partition_map` 写入 `run.json`，方便以后比较不同实验。
第四阶段将实际 p/k/q 写入 `identity` 并校验双方指纹，同时记录两边实际加载的层号。

第二阶段旧的 `partition: {"p": ..., "k": ..., "q": ...}` 仅保留兼容读取。
新增配置请用 `split`；不要同时写两种格式。以后换总层数不同的模型，
新格式无需手工修正 k。
