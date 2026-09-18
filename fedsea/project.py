"""Project-relative files and shared inference settings."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# utf-8-sig 同时兼容普通 UTF-8 和部分 Windows 编辑器写入的 BOM。
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def project_path(value, root=ROOT):
    # 配置里的相对路径统一以项目根目录为准，不随终端当前目录变化。
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Shared file paths must stay inside the project.")
    return path


def load_runtime_config(stage_config):
    # 只在内存中合并“共用配置 + 阶段覆盖”，不会反向改写共用 JSON。
    if "runtime_config" not in stage_config:
        return dict(stage_config)
    config = read_json(project_path(stage_config["runtime_config"]))
    overrides = stage_config.get("runtime_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(config):
        raise ValueError("runtime_overrides must contain only known inference settings.")
    return {**config, **overrides}
