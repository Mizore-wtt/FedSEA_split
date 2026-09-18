"""Per-session perturbation at the existing RPC boundary; the shared model is unchanged."""

from settings import ROOT
from https_core.session import HttpsSession
from noise import build_noise


class NoisyRpc:
    def __init__(self, transport, noise):
        self.transport, self.noise = transport, noise

    def call(self, path, metadata, tensors=None):
        if path == "/v1/forward":
            if not isinstance(tensors, dict) or set(tensors) != {
                "hidden", "attention_mask", "position_ids", "cache_position"
            }:
                raise ValueError("Noise applies only to a valid A-output forward request.")
            # Do not mutate A's tensor, masks or caller dictionary. No seed/noise metadata is sent.
            tensors = {**tensors, "hidden": self.noise.apply(tensors["hidden"])}
        return self.transport.call(path, metadata, tensors)


class NoiseSession(HttpsSession):
    def __init__(self, model, config, seed):
        self.noise_config, self.seed = dict(config), seed
        super().__init__(model)

    def reset(self):
        super().reset()
        self.noise = build_noise(self.noise_config, seed=self.seed)
        # A cache owns its RPC endpoint, so independent sessions never share an RNG or policy.
        self.cache.rpc = NoisyRpc(self.cache.rpc, self.noise)

    def generate(self, inputs, generation_config, **kwargs):
        # No cross-turn reuse: noisy prefixes from another draw/sigma must never be reused.
        # Within this call, A/B/C still cache normally and only new tokens cross HTTPS.
        self.reset()
        return super().generate(inputs, generation_config, **kwargs)
