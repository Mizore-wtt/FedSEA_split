"""Small deterministic Qwen2/MoE models and the real wire format, without downloaded weights."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from settings import ROOT, read_json
from fedsea.partition import Partition
from https_core.engine import ServerState, build_client
from https_core.weights import RoleWeights
from https_core.protocol import pack, unpack

NET = {"max_context_tokens": 128, "max_sessions": 4, "session_ttl_seconds": 600,
       "max_body_bytes": 1024 * 1024, "timeout_seconds": 3}
IDS = torch.tensor([[5, 7, 9, 11]])
GEN = GenerationConfig(
    do_sample=False, use_cache=True, max_new_tokens=3, min_new_tokens=3,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
)


def tiny(moe=False):
    config_class, model_class = (Qwen3MoeConfig, Qwen3MoeForCausalLM) if moe else (Qwen2Config, Qwen2ForCausalLM)
    config = config_class(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=not moe,
        head_dim=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    )
    config._attn_implementation = "eager"
    return model_class(config).eval().requires_grad_(False)


class RecordingRpc:
    def __init__(self, state):
        self.state = state
        self.calls = []
        self.uploads = []
        self.stats = {"requests": 0, "sent_body_bytes": 0, "received_body_bytes": 0, "rpc_seconds": 0.0}

    def call(self, path, metadata, tensors=None):
        self.calls.append(path)
        body = pack(metadata, tensors)
        meta, data = unpack(body)
        if path == "/v1/forward":
            self.uploads.append((meta, {name: tensor.clone() for name, tensor in data.items()}))
        result, output = self.state.dispatch(path, meta, data)
        reply = pack(result, output)
        self.stats["requests"] += 1
        self.stats["sent_body_bytes"] += len(body)
        self.stats["received_body_bytes"] += len(reply)
        return unpack(reply)


def build(directory, reference, part=Partition(2, 2, 2)):
    left = RoleWeights(directory, reference.config, part, "client", torch.device("cpu"), torch.float32)
    right = RoleWeights(directory, reference.config, part, "server", torch.device("cpu"), torch.float32)
    identity = {"test": "stage5", "p": part.p, "k": part.k, "q": part.q}
    state = ServerState(right, identity, NET)
    rpc = RecordingRpc(state)
    return build_client(left, rpc, identity, 128), state, rpc


def inputs(ids=IDS):
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def runtime(model, state, rpc):
    stage = read_json(STAGE / "config.json")
    stage["evaluation"].update(sigmas=[0.0, 0.1], seeds=[42, 43])
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(str(value) for value in ids))
    return SimpleNamespace(
        model=model, rpc=rpc, device=torch.device("cpu"), generation_config=GEN,
        tokenizer=tokenizer, synchronize=lambda: None, tokenize=lambda messages: inputs(),
        config={"system_prompt": "Test", "max_new_tokens": 3}, stage=stage,
        identity=state.identity, health={"weights": state.weights.describe()},
        profile=SimpleNamespace(name="tiny", describe=lambda: {"name": "tiny"}),
    )
