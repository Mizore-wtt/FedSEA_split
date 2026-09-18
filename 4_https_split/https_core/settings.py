"""Shared stage selection; describing a configuration never loads weights."""

import argparse
import copy
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "4_https_split"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedsea.model_registry import ModelRegistry
from fedsea.partition import resolve_partition
from fedsea.project import load_runtime_config, project_path, read_json
from fedsea.runtime import validate_config


def validate_network(net):
    if net["host"] != "127.0.0.1":
        raise ValueError("Stage 4 binds IPv4 loopback only, never a public/LAN interface.")
    for field, lower, upper in (
        ("port", 1, 65535), ("timeout_seconds", 1, 300),
        ("max_body_bytes", 1024, 64 * 1024**2), ("max_context_tokens", 1, 4096),
        ("max_sessions", 1, 8), ("session_ttl_seconds", 1, 3600),
    ):
        if type(net[field]) is not int or not lower <= net[field] <= upper:
            raise ValueError(f"Invalid HTTPS setting: {field}.")
    address = urlsplit(net["url"])
    if (address.scheme != "https" or address.hostname not in ("localhost", "127.0.0.1")
            or address.port != net["port"] or address.path not in ("", "/")
            or address.username or address.password or address.query or address.fragment):
        raise ValueError("Use an HTTPS loopback URL with the configured port, without extra components.")
    directory = project_path(net["credentials_dir"])
    if not directory.is_relative_to(ROOT / "certs") or directory == ROOT / "certs":
        raise ValueError("Credentials need a private subdirectory under certs/.")


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=STAGE / "config.json")
    result.add_argument("--model")
    result.add_argument("--split", dest="preset")
    result.add_argument("--p", type=int)
    result.add_argument("--q", type=int)
    result.add_argument("--device", choices=("cuda", "cpu"))
    result.add_argument("--max-new-tokens", type=int)
    result.add_argument("--port", type=int)
    result.add_argument("--describe", action="store_true")
    result.add_argument("--list-models", action="store_true")
    result.add_argument("--list-splits", action="store_true")
    return result


def select(args, *, stage_config=None, describe_extra=None):
    registry = ModelRegistry()
    if args.list_models:
        registry.list_models()
        return None
    if args.list_splits:
        print(json.dumps(read_json(ROOT / "configs/splits.json"), indent=2))
        return None
    stage = read_json(args.config) if stage_config is None else copy.deepcopy(stage_config)
    profile = registry.get(args.model or stage.get("model"))
    part = resolve_partition(
        profile.spec["num_hidden_layers"], stage.get("split"),
        preset=args.preset, p=args.p, q=args.q,
    )
    runtime = load_runtime_config(stage)
    if args.device:
        runtime["device"] = args.device
    if args.max_new_tokens is not None:
        runtime["max_new_tokens"] = args.max_new_tokens
    if args.port is not None:
        stage["https"]["port"] = args.port
        stage["https"]["url"] = f"https://localhost:{args.port}"
    validate_config(runtime)
    validate_network(stage["https"])
    if runtime["max_input_tokens"] + runtime["max_new_tokens"] > stage["https"]["max_context_tokens"]:
        raise ValueError("HTTPS context limit must cover max_input_tokens + max_new_tokens.")
    if args.describe:
        print(json.dumps({
            "model": profile.describe(), "partition": {"p": part.p, "k": part.k, "q": part.q},
            "partition_map": part.describe(), "https": stage["https"],
            "loads_weights": False, "opens_network": False, "noise_std": 0.0,
            **(describe_extra or {}),
        }, ensure_ascii=False, indent=2))
        return None
    profile.require_enabled()
    return stage, runtime, profile, part
