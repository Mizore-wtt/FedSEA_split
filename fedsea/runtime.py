"""Shared, offline, single-device inference runtime for all implemented stages."""

import importlib.metadata
import os
import platform
import sys
import time

import torch

from .model_registry import ModelRegistry
from .project import ROOT, project_path, read_json

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# 检查配置
def validate_config(config):
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if config["dtype"] not in ("float16", "bfloat16", "float32"):
        raise ValueError("dtype must be float16, bfloat16 or float32.")
    if config["attention_implementation"] != "eager":
        raise ValueError("The verified inference/comparison path currently requires eager attention.")
    for field in ("max_input_tokens", "max_new_tokens"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if not isinstance(config["system_prompt"], str):
        raise ValueError("system_prompt must be a string.")


# 检查上下文长度
def validate_length(input_tokens, config, model_context):
    if input_tokens > config["max_input_tokens"]:
        raise ValueError(
            f"Input has {input_tokens} tokens; limit is {config['max_input_tokens']}. "
            "Shorten the input or use /reset. Nothing has been silently truncated."
        )
    if input_tokens + config["max_new_tokens"] > model_context:
        raise ValueError("Input plus generation budget exceeds the model context window.")


# 组装提示词
def make_messages(prompt, system_prompt):
    if not prompt.strip():
        raise ValueError("The prompt must not be empty.")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def estimated_weight_bytes(profile, dtype):
    return profile.settings["parameter_count"] * (4 if dtype == "float32" else 2)


class ModelRuntime:
    def __init__(self, config, model_name=None):
        validate_config(config)
        self.profile = ModelRegistry().get(model_name)
        self.profile.require_enabled()
        self.spec = self.profile.spec
        self.config = dict(config)
        model_dir = project_path(self.spec["local_dir"])
        manifest_path = model_dir / "_fedsea_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Model '{self.profile.name}' is not prepared. Run scripts/prepare_model.py "
                "with the same --model selection."
            )
        manifest = read_json(manifest_path)
        if any(manifest.get(key) != self.spec[key] for key in ("repo_id", "revision")):
            raise ValueError("Prepared model identity differs from the selected profile.")
        actual_config = read_json(model_dir / "config.json")
        for key in ("model_type", "num_hidden_layers", "hidden_size"):
            if actual_config.get(key) != self.spec[key]:
                raise ValueError(f"Model profile does not match the actual checkpoint: {key}.")
        if self.profile.settings["adapter"] != self.spec["model_type"]:
            raise ValueError("The adapter key must match the checkpoint model_type.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        # 检查判断设备
        self.torch = torch
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use the project .venv, or explicitly select --device cpu.")
        self.device = torch.device(config["device"])
        self.dtype = torch.float32 if self.device.type == "cpu" else getattr(torch, config["dtype"])
        if self.device.type == "cuda":
            if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support the requested bfloat16 dtype.")
            free, _ = torch.cuda.mem_get_info(self.device)
            minimum = estimated_weight_bytes(self.profile, config["dtype"])
            if minimum + 512 * 1024**2 > free:
                raise RuntimeError(
                    f"Single-GPU loading needs about {minimum / 1024**3:.1f} GiB for weights alone; "
                    f"only {free / 1024**3:.1f} GiB is free. Offloading/multi-GPU/quantization "
                    "is not implemented. See docs/models.md."
                )
        torch.manual_seed(config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(self.device)
        # tokenizer 把文字转换为 token ID；模型权重只从本地目录读取。
        print(f"Loading {self.profile.name} locally on {self.device} ({self.dtype})...", flush=True)
        start = time.perf_counter()  # 计时
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=self.dtype, attn_implementation=config["attention_implementation"],
            local_files_only=True, trust_remote_code=False,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        # eval() 关闭训练行为；推理时还会用 inference_mode() 避免建立梯度计算图。
        self.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.generation_config = GenerationConfig(
            max_new_tokens=config["max_new_tokens"], do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id,
        )
        self.environment = {
            "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "huggingface-hub", "safetensors", "numpy")
            },
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "dtype": str(self.dtype),
            "model_load_seconds": self.load_seconds,
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        print(f"Loaded in {self.load_seconds:.2f}s. Offline inference; no noise.", flush=True)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def tokenize(self, messages):
        # 先按模型自己的对话模板拼接，再分词。返回 [batch, tokens]，不是隐藏向量。
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            **self.profile.template_kwargs,
        ).to(self.device)
        validate_length(inputs["input_ids"].shape[-1], self.config, self.model.config.max_position_embeddings)
        return inputs
