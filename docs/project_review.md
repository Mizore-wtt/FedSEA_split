# 项目复查记录

[项目入口](../README.md) | [第二阶段操作](../2_model_sep/README.md) | [代码阅读顺序](code_guide.md)

复查日期：2026-09-08。实际项目目录：`E:\0_wt_ws\FedSEA`。
保留现有阶段目录、模型接口与 p/q 切分接口，不重新下载权重，不改推理依赖或误差阈值。

## 第二阶段报错原因

原 `configs/model.json` 的 `baseline_run` 固定指向
`1_model_base/runs/20260907T023904Z_d49a4ec1`，该目录在本次复查时已不存在。
旧入口在开始本轮比较前直接读取它的 `run.json`，因此提前退出；
这不是模型权重丢失，也不是 A/B/C 层无法计算。

## 修改范围

- 将失效的默认 `baseline_run` 改为 `null`。主对比仍实际执行完整模型和 A/B/C，两边都关闭缓存。
- 新增 `2_model_sep/history.py`：自动参考缺失或报告损坏时明确跳过，显式选择的错误路径则在加载权重前报错。
- 新增 `--stage1-run none|auto|路径`，保存实际历史来源、读取摘要和跳过原因，不自动替换成其他历史实验。
- 比较异常或中断会记录为 `error` / `interrupted`；空测试和真实数值不一致不会被判为通过。
- 修复第三阶段裁剪缓存失败，以及第三、四阶段生成前同步失败时的会话/临时 hook 清理。
- 补充配置合并、A/B/C 位置与掩码、缓存槽位、HTTPS 传输和参数加载的关键注释。
- 新增 `scripts/run_tests.py` 统一测试入口及中文 `docs/code_guide.md`，更新启动说明和历史状态说明。

## 验证

以下均在正式项目目录执行，真实模型为本地 Qwen2.5-0.5B-Instruct，
设备为 RTX A3000 Laptop GPU，使用 FP16、eager attention、贪心解码，
生成上限仍为默认 128 token；没有降低长度覆盖或放宽 `1e-5` 误差阈值。

### 自动测试

执行 `.\.venv\Scripts\python.exe -B -X utf8 .\scripts\run_tests.py --report`：

| 测试组 | 通过数量 |
|---|---:|
| 共用接口和 CLI | 28 |
| 第一阶段 | 10 |
| 第二阶段 | 54 |
| 第三阶段 | 33 |
| 第四阶段 | 37 |
| 合计 | 162 |

全部通过，无跳过项。包括随机 Qwen2/Qwen3-MoE、真实回环 TLS 连接、
小型 CUDA 长位置 RoPE 测试，以及本次新增的缺失历史、数值不一致、
中断落盘、缓存裁剪失败、同步失败清理回归。
汇总和分组日志位于[自动测试报告目录](../tests/runs/20260908T050135Z_44f9297f/)。

### 真实小模型

| 检查 | 结果及报告 |
|---|---|
| 第一阶段 `--benchmark --save-artifacts` | 3 个问题正常生成，基本指令/算术检查通过；[报告](../1_model_base/runs/20260908T050305Z_77d5b422/run.json) |
| 原报错命令 `run_split.py --compare`，2/20/2 | 5 个问题 + 4 种输入长度全部通过，历史为 `skipped`；[报告](../2_model_sep/runs/20260908T050142Z_1a3b151b/run.json) |
| `run_split.py --compare --p 1 --q 3`，显式读取新基线 | 5 个问题 + 4 种长度全部通过；3 个基线问题历史 `passed`，额外 2 个为 `not_applicable`；[报告](../2_model_sep/runs/20260908T050351Z_1bfe359c/run.json) |
| 第三阶段 `run_cached.py --compare`，2/20/2 | 5 个问题 + 4 种长度 + 3 轮会话全部通过；[报告](../3_kv_cache/runs/20260908T050517Z_a03db7e7/run.json) |
| 第四阶段 `verify_local.py`，2/20/2 | 5 个问题 + 4 种长度 + 3 轮会话全部通过；[报告](../4_https_split/runs/20260908T050626Z_65c551b5/run.json) |

第二阶段两种切分的生成 token、回答和所有已检查 logits 完全相等，
输入长度覆盖 1、128、512、1024 token。历史跳过不影响实际执行的主对比。
第三阶段多轮复用检查中成功复用 40 个前缀 token，重置后的复用量为 0。
第四阶段使用不同的客户端/服务端进程，临时端口为 61082；
结束时服务端剩余会话为 0、测试服务已关闭，见[启动器记录](../4_https_split/runs/launcher_20260908T050614Z_7ab8abf7/launcher.json)。
这些结果是正确性回归，不用于速度结论。

此前各阶段 `VERIFICATION.md` 和 `docs/verification_interfaces.md` 保留为历史快照，
其中旧结果目录可能已被清理，不能代替本次验收记录。

## 使用与备份

原命令可以直接重新执行，不需要再跑第一阶段，也不需要重新安装环境：

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --compare
```

优先读项目 README 了解启动，读第二阶段 README 了解比较状态，
读 `docs/code_guide.md` 按函数顺序理解代码。
新增结果仍各归各的阶段目录；模型、环境、凭据没有移动或重复复制。
首次应用修改前的源码备份位于
`E:\0_wt_ws\pre\tmp\fedsea_review_backup_20260908_130109_159`，
不包含模型、私钥或已有实验结果。

## 未改变的边界

目前是推理实验，没有训练、联邦聚合或加噪。HTTPS 仍仅供同机回环双进程验证。
Qwen3-30B 只保留配置与架构适配接口，没有下载、部署或验收真实 30B 权重。
比较通过不证明回答事实正确，也不证明获得隐私保证或推理加速。
