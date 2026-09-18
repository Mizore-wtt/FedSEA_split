"""Read only owned safetensors parameters; no complete model is constructed."""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import transformers
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2 import modeling_qwen2 as q2
from transformers.models.qwen3_moe import modeling_qwen3_moe as q3

from .settings import ROOT
from fedsea.checkpoints import checkpoint_files, sha256
from fedsea.project import project_path, read_json

ADAPTERS = {
    "qwen2": (q2.Qwen2DecoderLayer, q2.Qwen2RMSNorm, q2.Qwen2RotaryEmbedding),
    "qwen3_moe": (q3.Qwen3MoeDecoderLayer, q3.Qwen3MoeRMSNorm, q3.Qwen3MoeRotaryEmbedding),
}
TRANSFORMERS_VERSION = "4.56.1"


def validate_architecture(config, part):
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"HTTPS adapters require transformers=={TRANSFORMERS_VERSION}.")
    part.validate(config.num_hidden_layers)
    if config.model_type not in ADAPTERS:
        raise ValueError("No HTTPS weight adapter for this architecture.")
    if config._attn_implementation != "eager":
        raise ValueError("The verified path requires eager attention.")
    if getattr(config, "sliding_window", None) is not None or any(
        kind != "full_attention" for kind in getattr(config, "layer_types", [])
    ):
        raise ValueError("Only full attention is supported.")
    rope = getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear", "yarn"):
        raise ValueError("Stateful/dynamic RoPE is not verified across independently owned stages.")


class RoleWeights(nn.Module):
    def __init__(self, directory, config, part, role, device, dtype):
        super().__init__()
        if role not in ("client", "server"):
            raise ValueError("role must be client or server.")
        validate_architecture(config, part)
        self.config, self.partition, self.role = copy.deepcopy(config), part, role
        decoder, norm, rotary = ADAPTERS[config.model_type]
        self.indices = (
            list(range(part.p)) + list(range(part.p + part.k, config.num_hidden_layers))
            if role == "client" else list(range(part.p, part.p + part.k))
        )
        # Meta construction allocates no weights for owned modules, or for unowned layers.
        with torch.device("meta"):
            self.layers = nn.ModuleDict({str(i): decoder(config, i) for i in self.indices})
            if role == "client":
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
                self.norm = norm(config.hidden_size, eps=config.rms_norm_eps)
                if not config.tie_word_embeddings:
                    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        paths = checkpoint_files(directory)
        sources = {}
        for filename in paths:
            if filename.endswith(".safetensors"):
                with safe_open(Path(directory) / filename, framework="pt", device="cpu") as source:
                    for name in source.keys():
                        if name in sources:
                            raise ValueError(f"Duplicate checkpoint parameter: {name}.")
                        sources[name] = filename
        if "model.safetensors.index.json" in paths:
            index = read_json(Path(directory) / "model.safetensors.index.json")["weight_map"]
            if index != sources:
                raise ValueError("Checkpoint shard contents differ from their index.")
        expected = self.state_dict()
        grouped = {}
        for name, tensor in expected.items():
            upstream = name if name.startswith("lm_head.") else "model." + name
            if upstream not in sources:
                raise ValueError(f"Missing owned parameter: {upstream}.")
            grouped.setdefault(sources[upstream], []).append((name, upstream, tensor.shape))
        state = {}
        # Open each needed shard once and fetch only this role's parameter keys, not the full state.
        for filename, entries in grouped.items():
            with safe_open(Path(directory) / filename, framework="pt", device="cpu") as source:
                for name, upstream, shape in entries:
                    tensor = source.get_tensor(upstream)
                    if tensor.shape != shape or not tensor.is_floating_point():
                        raise ValueError(f"Invalid shape/dtype: {upstream}.")
                    state[name] = tensor.to(device=device, dtype=dtype, copy=True)
        self.load_state_dict(state, strict=True, assign=True)
        if role == "client" and config.tie_word_embeddings:
            # Preserve tied embedding/head ownership: both names must point at one Parameter.
            if "lm_head.weight" in sources:
                with safe_open(Path(directory) / sources["lm_head.weight"], framework="pt", device="cpu") as src:
                    head = src.get_tensor("lm_head.weight").to(device=device, dtype=dtype)
                    if not torch.equal(head, self.embed_tokens.weight):
                        raise ValueError("Checkpoint declares tied embeddings but stores different head weights.")
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta")
            self.lm_head.weight = self.embed_tokens.weight
        # Match from_pretrained(): initialize frequencies on CPU, then move without casting.
        # GPU-side pow initialization differs slightly and changes FP16 rounding at long positions.
        self.rotary_emb = rotary(config, device="cpu").to(device=device)
        self.eval().requires_grad_(False)
        self.loaded_keys = sorted(name for entries in grouped.values() for _, name, _ in entries)
        self.parameter_count = sum(t.numel() for t in self.parameters())
        self.weight_bytes = sum(t.numel() * t.element_size() for t in self.parameters())
        if any(t.is_meta for t in self.parameters()):
            raise RuntimeError("An owned parameter was left uninitialized.")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def run(self, hidden, context, cache, indices=None):
        for i in self.indices if indices is None else indices:
            hidden = self.layers[str(i)](hidden, **context, past_key_values=cache, use_cache=True)
        return hidden

    def describe(self, *, include_keys=False):
        result = {
            "role": self.role, "layers_1based": [i + 1 for i in self.indices],
            "parameters": self.parameter_count, "weight_bytes": self.weight_bytes,
            "full_model_loaded": False,
        }
        if include_keys:
            result["loaded_parameter_keys"] = self.loaded_keys
        return result


def prepare(profile, part, runtime):
    spec = profile.spec
    directory = project_path(spec["local_dir"])
    manifest = read_json(directory / "_fedsea_manifest.json")
    if any(manifest.get(k) != spec[k] for k in ("repo_id", "revision")):
        raise ValueError("Prepared model identity differs from the selected profile.")
    files = ["config.json", *checkpoint_files(directory)]
    hashes = {name: sha256(directory / name) for name in files}
    for name, digest in hashes.items():
        if manifest.get("files", {}).get(name, {}).get("sha256") != digest:
            raise ValueError(f"Checkpoint integrity failure: {name}. Re-prepare the model.")
    if spec.get("weights_sha256") and hashes.get("model.safetensors") != spec["weights_sha256"]:
        raise ValueError("Weight checksum differs from the pinned model profile.")
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "eager"
    config.use_cache = True
    if hasattr(config, "output_router_logits"):
        config.output_router_logits = False
    for name in ("model_type", "num_hidden_layers", "hidden_size"):
        if getattr(config, name) != spec[name]:
            raise ValueError(f"Model profile mismatch: {name}.")
    if profile.settings["adapter"] != config.model_type:
        raise ValueError("Adapter/profile mismatch.")
    validate_architecture(config, part)
    device = torch.device(runtime["device"])
    dtype = torch.float32 if device.type == "cpu" else getattr(torch, runtime["dtype"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the project Python, or --device cpu on BOTH sides.")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16 is unsupported by this GPU.")
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(runtime["seed"])
    identity = {
        "protocol": 1, "repo_id": spec["repo_id"], "revision": spec["revision"],
        "files_sha256": hashes, "model_type": config.model_type, "dtype": str(dtype),
        "transformers": TRANSFORMERS_VERSION, "attention": "eager",
        "p": part.p, "k": part.k, "q": part.q,
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return directory, config, device, dtype, identity
