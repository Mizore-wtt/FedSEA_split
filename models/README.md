# 共用模型目录

[返回项目入口](../README.md) | [准备模型](../docs/setup.md) | [模型配置](../docs/models.md)

每个模型单独一个目录，所有阶段共用，不在阶段内复制权重。
目前只部署了 `Qwen2.5-0.5B-Instruct/`；Qwen3 的配置预留不代表这里已有权重。

仓库名称、固定 revision、目录和架构由 `configs/model.json` 管理。
新增模型通过 `scripts/prepare_model.py --model <模型名>` 准备，支持单文件与
索引分片 safetensors。每个目录保留模型/分词器配置、权重、
`_fedsea_manifest.json`，以及来源中存在的 LICENSE 和上游模型说明。

不要手工改上游 `config.json` 来改变 p/q，应修改项目的切分配置。
这些权重只读复用，正常推理不自动下载。权重目录不进入版本库。
SHA-256 是完整性检查；本地导入的可信来源与版本仍需确认。
