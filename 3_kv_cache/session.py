"""Single-conversation cache reuse based on exact token and attention-mask prefixes."""

import copy
import time

import torch
from transformers.cache_utils import DynamicCache


class CachedSession:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        factory = getattr(self.model, "new_cache", None)
        self.cache = factory() if factory else DynamicCache(config=self.model.config)
        self.cached_ids = None
        self.cached_mask = None

    def generate(self, inputs, generation_config, *, trace=False, synchronize=lambda: None):
        # Prefix cropping and generation are one transaction: either both succeed, or reset.
        try:
            return self._generate(inputs, generation_config, trace, synchronize)
        except BaseException:
            self.reset()
            raise

    def _generate(self, inputs, generation_config, trace, synchronize):
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise ValueError("Session inputs support input_ids and attention_mask only.")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or mask.shape != ids.shape:
            raise ValueError("Sessions require one nonempty conversation and a full 2D mask.")
        if (generation_config.use_cache is False or generation_config.do_sample
                or generation_config.num_beams != 1):
            raise ValueError("Sessions require cache-enabled greedy decoding.")
        maximum = generation_config.max_new_tokens
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("Set a positive max_new_tokens for this session.")
        if ids.shape[1] + maximum > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus generation budget exceeds the context window.")
        reuse = 0
        if self.cached_ids is not None:
            if self.cache.get_seq_length() != self.cached_ids.shape[1]:
                self.reset()
                raise ValueError("Session token history no longer matches its cache; session reset.")
            common = min(self.cached_ids.shape[1], ids.shape[1])
            equal = (self.cached_ids[:, :common] == ids[:, :common]) & (
                self.cached_mask[:, :common] == mask[:, :common]
            )
            mismatch = (~equal[0]).nonzero()
            reuse = int(mismatch[0].item()) if len(mismatch) else common
            # HF generate() must evaluate at least the final prompt token for next-token logits.
            reuse = min(reuse, ids.shape[1] - 1)
            if reuse == 0:
                self.reset()
            elif reuse < self.cache.get_seq_length():
                self.cache.crop(reuse)
        query_lengths = []

        def record_input(_model, args, kwargs):
            tokens = kwargs.get("input_ids", args[0] if args else None)
            if tokens is not None:
                query_lengths.append(tokens.shape[1])

        handle = self.model.register_forward_pre_hook(record_input, with_kwargs=True)
        try:
            # CUDA can surface an earlier failure here; the hook must still be removed.
            synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, past_key_values=self.cache,
                    generation_config=copy.deepcopy(generation_config),
                    return_dict_in_generate=True, output_logits=trace,
                )
            synchronize()
            seconds = time.perf_counter() - start
            self.cache = output.past_key_values
            cache_length = self.cache.get_seq_length()
            sequence = output.sequences
            if cache_length != sequence.shape[1] - 1:
                raise RuntimeError("Unexpected final cache length for greedy generation.")
            self.cached_ids = sequence[:, :cache_length].detach().clone()
            added = torch.ones(
                1, sequence.shape[1] - ids.shape[1], dtype=mask.dtype, device=mask.device
            )
            self.cached_mask = torch.cat((mask, added), dim=1)[:, :cache_length].detach().clone()
        finally:
            handle.remove()
        return output, {
            "input_tokens": ids.shape[1], "reused_prefix_tokens": reuse,
            "prefill_tokens": ids.shape[1] - reuse, "cached_tokens": cache_length,
            "generated_tokens_including_eos": sequence.shape[1] - ids.shape[1],
            "query_tokens_per_forward": query_lengths,
            "generation_seconds": seconds,
            "timing_note": "Includes cache bookkeeping; trace mode also collects logits. Not a speed benchmark.",
        }
