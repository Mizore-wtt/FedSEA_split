"""Validate single-file or indexed safetensors checkpoints before copying."""

import hashlib
from pathlib import Path

from .project import read_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(source):
    source = Path(source)
    single, index = source / "model.safetensors", source / "model.safetensors.index.json"
    if single.exists() and index.exists():
        raise ValueError("Ambiguous checkpoint: both single-file and sharded weights are present.")
    if index.is_file():
        weight_map = read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The safetensors index needs a nonempty weight_map.")
        shards = set()
        for filename in weight_map.values():
            if (not isinstance(filename, str) or "/" in filename or "\\" in filename
                    or ":" in filename or Path(filename).name != filename
                    or not filename.endswith(".safetensors")):
                raise ValueError("Shard names must be plain safetensors filenames, not paths.")
            shards.add(filename)
        missing = [name for name in sorted(shards) if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
        return ["model.safetensors.index.json", *sorted(shards)]
    if single.is_file():
        return ["model.safetensors"]
    raise FileNotFoundError("No safetensors checkpoint found; pickle/.bin loading is not supported.")


def snapshot_files(source):
    source = Path(source)
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model/tokenizer metadata: {missing}")
    optional = [
        "generation_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
        "LICENSE", "README.md",
    ]
    return required + [name for name in optional if (source / name).is_file()] + checkpoint_files(source)
