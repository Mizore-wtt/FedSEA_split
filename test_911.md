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
分布式推理、机密推理
安全加固
```bash
.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --chat --sigma 0.02

.\.venv\Scripts\python.exe -X utf8 .\1_model_base\baseline.py --chat

.\.venv\Scripts\python.exe -X utf8 .\2_model_sep\run_split.py --chat

.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --chat --sigma 0.5

.\.venv\Scripts\python.exe -X utf8 .\5_noise_eval\run_local.py --evaluate
```

| Sigma | Trials | Task Accuracy | Reference Match | Mean Effective Noise RMS |
|---:|---:|---:|---:|---:|
| 0 | 12 | 1.000 | 1.000 | 0.000000 |
| 0.02 | 12 | 1.000 | 0.833 | 0.020033 |
| 0.05 | 12 | 1.000 | 0.833 | 0.050086 |
| 0.1 | 12 | 1.000 | 0.750 | 0.100174 |

现状、思路、效果对比、对比数据（
缓存命中率、安全