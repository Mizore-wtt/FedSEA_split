"""Experimental A-output perturbations. No DP accounting or cryptographic randomness."""

import hashlib
import math
import time

import torch


def validate_sigma(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("sigma must be a finite, nonnegative standard deviation.")


def validate_seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Noise seeds must be integers in [0, 2**63).")


def derive_seed(seed, case_id):
    """Stable per-case streams; the same case/seed pair is shared across sigma settings."""
    validate_seed(seed)
    digest = hashlib.sha256(f"fedsea-noise-v1\0{seed}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


class GaussianNoise:
    def __init__(self, sigma, seed):
        validate_sigma(sigma)
        validate_seed(seed)
        self.sigma, self.seed = float(sigma), seed
        # A private CPU generator does not reseed model inference or other sessions.
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.calls = self.elements = self.changed = 0
        self.signal_squared = self.noise_squared = self.noise_absolute = 0.0
        self.max_delta = self.seconds = 0.0

    def apply(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] != 1 or min(hidden.shape) < 1
                or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(hidden).all()):
            raise ValueError("Noise input must be finite FP16/BF16/FP32 [1, new_tokens, hidden].")
        self.calls += 1
        self.elements += hidden.numel()
        # Exact identity, including dtype/storage; zero must not consume random numbers.
        if self.sigma == 0:
            return hidden
        start = time.perf_counter()
        sample = torch.randn(hidden.shape, generator=self.generator, dtype=torch.float32, device="cpu")
        perturbed = (hidden.float() + sample.to(hidden.device) * self.sigma).to(hidden.dtype)
        if not torch.isfinite(perturbed).all():
            raise ValueError("Noise caused a nonfinite activation; reduce sigma. Nothing was uploaded.")
        # Measure the effective perturbation AFTER rounding to the transmitted model dtype.
        original = hidden.double()
        delta = perturbed.double() - original
        self.signal_squared += original.square().sum().item()
        self.noise_squared += delta.square().sum().item()
        self.noise_absolute += delta.abs().sum().item()
        self.max_delta = max(self.max_delta, delta.abs().max().item())
        self.changed += torch.count_nonzero(delta).item()
        self.seconds += time.perf_counter() - start
        return perturbed

    def describe(self):
        count = self.elements
        rms = math.sqrt(self.noise_squared / count) if count else 0.0
        signal = math.sqrt(self.signal_squared / count) if count and self.sigma else None
        return {
            "distribution": "gaussian", "sigma": self.sigma, "effective_seed": self.seed,
            "sigma_definition": "absolute per-element standard deviation before dtype rounding",
            "random_source": "private torch.Generator(cpu), float32 normal samples",
            "forward_calls": self.calls, "elements": count,
            "effective_noise_rms": rms,
            "effective_mean_abs_noise": self.noise_absolute / count if count else 0.0,
            "effective_max_abs_noise": self.max_delta,
            "changed_element_fraction": self.changed / count if count else 0.0,
            "clean_activation_rms": signal,
            "snr_db": 20 * math.log10(signal / rms) if signal and rms else None,
            "transform_seconds_including_statistics": self.seconds,
        }


NOISE_POLICIES = {"gaussian": GaussianNoise}


def validate_noise(config):
    if not isinstance(config, dict) or set(config) != {"distribution", "sigma", "seed"}:
        raise ValueError("noise requires exactly distribution, sigma, seed.")
    if not isinstance(config["distribution"], str) or config["distribution"] not in NOISE_POLICIES:
        raise ValueError("Unsupported noise distribution; add and verify a NOISE_POLICIES adapter first.")
    validate_sigma(config["sigma"])
    validate_seed(config["seed"])


def build_noise(config, *, seed=None):
    validate_noise(config)
    return NOISE_POLICIES[config["distribution"]](config["sigma"], config["seed"] if seed is None else seed)
