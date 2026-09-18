"""Prepare a selected, pinned model from a local snapshot or the official HTTPS hub."""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedsea.checkpoints import sha256, snapshot_files
from fedsea.model_registry import ModelRegistry
from fedsea.project import project_path, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Profile name from configs/model.json.")
    parser.add_argument("--source", type=Path, help="Existing Hugging Face snapshot directory.")
    parser.add_argument("--list-models", action="store_true", help="Only list profiles.")
    args = parser.parse_args()
    registry = ModelRegistry()
    if args.list_models:
        registry.list_models()
        return
    profile = registry.get(args.model)
    profile.require_enabled()
    spec = profile.spec
    destination = project_path(spec["local_dir"])
    if args.source:
        source = args.source.resolve(strict=True)
        origin = "local_snapshot"
    else:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(
            repo_id=spec["repo_id"], revision=spec["revision"],
            cache_dir=ROOT / ".cache/huggingface",
            allow_patterns=[
                "*.json", "*.safetensors", "chat_template.jinja", "merges.txt", "LICENSE", "README.md"
            ],
            endpoint="https://huggingface.co",
        ))
        origin = "official_https_download"
    files = snapshot_files(source)
    config = read_json(source / "config.json")
    for field in ("model_type", "num_hidden_layers", "hidden_size"):
        if config.get(field) != spec[field]:
            raise ValueError(f"Unexpected model architecture: {field}={config.get(field)}")
    print("Checking source file SHA-256 digests...", flush=True)
    digests = {name: sha256(source / name) for name in files}
    if spec.get("weights_sha256") and digests.get("model.safetensors") != spec["weights_sha256"]:
        raise ValueError("The weight digest does not match the pinned model.")
    for old_file in destination.glob("*.safetensors*"):
        if old_file.name not in files:
            raise FileExistsError(f"Refusing to mix old and new checkpoint layouts: {old_file}")
    for name in files:
        target = destination / name
        if target.exists() and sha256(target) != digests[name]:
            raise FileExistsError(f"Refusing to overwrite a different file: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in files:
        target = destination / name
        if not target.exists():
            shutil.copy2(source / name, target)
        if sha256(target) != digests[name]:
            raise OSError(f"Copy verification failed: {target}")
        records[name] = {"bytes": target.stat().st_size, "sha256": digests[name]}
        print(f"Verified {name}", flush=True)
    manifest = {
        "repo_id": spec["repo_id"], "revision": spec["revision"], "profile": profile.name,
        "origin": origin, "source_path": str(source),
        "installed_at_utc": datetime.now(timezone.utc).isoformat(), "files": records,
    }
    (destination / "_fedsea_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Model ready: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
