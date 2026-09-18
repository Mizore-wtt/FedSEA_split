# 环境与模型准备

[返回项目入口](../README.md)

**当前电脑已经部署好小模型，日常使用无需执行安装命令。**
本页用于换电脑、环境损坏或新增模型。

## 重建环境

项目实际路径为 `E:\0_wt_ws\FedSEA`。`.venv` 只服务于本项目，
不修改已有 Conda 环境。它依赖创建时的基础 Python，不能直接搬到其他电脑使用。
以下命令需要已安装 `uv`，并能获取 Python 3.11 和软件包：

```powershell
Set-Location 'E:\0_wt_ws\FedSEA'
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements\cuda-cu126.txt
uv pip install --python .venv\Scripts\python.exe -r requirements\base.txt
uv pip check --python .venv\Scripts\python.exe
```

当前验证环境为 Python 3.11.15、PyTorch 2.8.0+cu126、Transformers 4.56.1。
`requirements/windows-lock.txt` 保存本机完整包版本；精确重建时先装 CUDA
依赖，再用该锁定文件替代最后的 `base.txt` 安装步骤。
不要为了换模型直接升级 Transformers，第二、三、四阶段依赖其具体内部接口。

## 准备模型

权重按模型分目录，模型名称、固定 revision、本地目录统一在
`configs/model.json` 中登记。当前默认模型已经就绪。

```powershell
# 查看已登记模型，不联网、不加载权重
.\.venv\Scripts\python.exe -X utf8 .\scripts\prepare_model.py --list-models
# 从可信的已有 Hugging Face snapshot 复制到本项目
.\.venv\Scripts\python.exe -X utf8 .\scripts\prepare_model.py --model qwen2.5-0.5b-instruct --source '你的snapshot绝对路径'
# 没有本地 snapshot 时，按固定 revision 从官方 HTTPS 下载
.\.venv\Scripts\python.exe -X utf8 .\scripts\prepare_model.py --model qwen2.5-0.5b-instruct
```

脚本支持单个 `model.safetensors`，或索引 `model.safetensors.index.json`
及其所有分片。缺失分片、路径异常、模型结构不符会报错；不覆盖内容不同的
已有文件，不混用两个模型的权重。复制后逐文件校验 SHA-256 并生成
`_fedsea_manifest.json`。模型说明、LICENSE 等上游附属文件存在时也会保留。

当前小模型还固定了预期权重 SHA-256。其他模型若只记录复制摘要，
只能证明复制前后一致，不能单凭该摘要确认本地来源的真实性或 revision。
本地导入请确认 snapshot 来自配置中的固定版本。
第 1–3 阶段正常加载检查 manifest 身份和模型结构，不会每次重算全部权重摘要。
第 4 阶段启动还会重算权重/配置摘要，用于严格确认两个进程选择的是同一份模型。

未启用的 Qwen3 配置会直接拒绝下载/加载；启用前按[模型说明](models.md)
完成版本选择、硬件规划和配置检查，不要只把 `enabled` 改成 `true`。

## 启动排错

| 现象 | 处理 |
|---|---|
| `.venv` 中找不到 Python | 检查工作路径；换电脑后需要重建环境 |
| PowerShell 禁止 `.ps1` | 用 README 中的 `.venv\Scripts\python.exe` 命令 |
| 中文终端输出异常 | 保留启动参数 `-X utf8` |
| `CUDA unavailable` | 确认使用项目解释器；可明确加 `--device cpu` |
| `not prepared` | 检查所选模型的目录与 manifest，必要时运行准备脚本 |
| `reserved` | 当前只是预留模型；用 `--describe` 预览，不会加载 |
| 输入过长 | `/reset` 或缩短输入；不会静默截断历史 |
| 显存预检查失败 / OOM | 先停止其他占显存任务或规划硬件；没有自动卸载/量化 |
| 第二阶段找不到旧 runs 的 run.json | 更新后的默认主对比不依赖旧结果；显式历史路径请修正，或用 `--stage1-run none` |

安装和下载需要联网；第 1–3 阶段正常推理只从本地加载，禁用远程自定义代码。
此时不需要证书，也没有浏览器页面或 HTTPS 服务。

## 全部轻量测试

在项目根目录依次执行，均不需要加载真实模型权重：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --report
# 也可只选一套或几套
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_tests.py --suite 2 3
```

共用接口与第 1–5 阶段各在独立进程执行；`--report` 保存到 `tests/runs/`。
第四、五阶段使用临时证书和回环连接，本机有 CUDA 时还运行相应的小型 GPU 回归。
测试不下载模型、不修改日常证书；它们不等于真实模型和真实跨机器部署验收。

真实小模型检查再运行第一阶段 `--benchmark --save-artifacts`、第二阶段
`run_split.py --compare`、第三阶段 `run_cached.py --compare`，
以及证书已准备好后的第四阶段 `verify_local.py`（自动创建和关闭自己的测试服务）。
不要把随机小模型的测试通过当作真实大模型已部署。
第五阶段真实加噪试验使用 `5_noise_eval/run_local.py --evaluate`，默认也会自动关闭自己的服务。
