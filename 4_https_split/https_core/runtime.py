"""Normal clients load A/C only. A full model is loaded only by --compare."""

import torch
from transformers import AutoTokenizer, GenerationConfig

from .settings import project_path
from .transport import HttpsRpc
from .weights import RoleWeights, prepare
from .engine import build_client
from fedsea.runtime import validate_length


class ClientRuntime:
    def __init__(self, stage, runtime, profile, part, *, boundary_label="noise OFF"):
        self.stage, self.config, self.profile, self.partition = stage, runtime, profile, part
        self.directory, config, self.device, self.dtype, self.identity = prepare(profile, part, runtime)
        net = stage["https"]
        self.rpc = HttpsRpc(
            net["url"], project_path(net["credentials_dir"]), net["timeout_seconds"], net["max_body_bytes"]
        )
        self.health, tensors = self.rpc.call("/v1/health", {})
        if tensors or self.health.get("identity") != self.identity:
            raise ValueError("Server model/split/precision differs. Start both sides with the SAME options.")
        if (self.health["max_context_tokens"] != net["max_context_tokens"]
                or self.health["max_body_bytes"] != net["max_body_bytes"]):
            raise ValueError("Client/server HTTPS limits differ.")
        print("Verified server certificate, identity, and partition. Loading A/C only...", flush=True)
        weights = RoleWeights(self.directory, config, part, "client", self.device, self.dtype)
        self.model = build_client(weights, self.rpc, self.identity, net["max_context_tokens"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory, local_files_only=True, trust_remote_code=False
        )
        upstream = GenerationConfig.from_pretrained(self.directory, local_files_only=True)
        self.generation_config = GenerationConfig(
            max_new_tokens=runtime["max_new_tokens"], do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=upstream.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        print(f"Client ready: A={part.p}, B=HTTPS({part.k}), C={part.q}; cache ON; {boundary_label}.", flush=True)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def tokenize(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            **self.profile.template_kwargs,
        ).to(self.device)
        validate_length(inputs["input_ids"].shape[1], self.config,
                        min(self.model.config.max_position_embeddings, self.stage["https"]["max_context_tokens"]))
        return inputs
