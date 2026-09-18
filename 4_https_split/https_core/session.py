"""One local conversation, with exact-prefix reuse and remote cache cleanup."""

import copy
import time

import torch


class HttpsSession:
    def __init__(self, model):
        self.model, self.cache = model, None
        self.reset()

    def close(self):
        if self.cache is not None:
            self.cache.close()
        self.cached_ids = self.cached_mask = None

    def reset(self):
        self.close()
        self.cache = self.model.new_cache()

    def generate(self, inputs, generation_config, *, trace=False, synchronize=lambda: None):
        try:
            return self._generate(inputs, generation_config, trace, synchronize)
        except BaseException:
            self.reset()
            raise

    def _generate(self, inputs, generation_config, trace, synchronize):
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise ValueError("Sessions accept input_ids and attention_mask only.")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if (ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] < 1 or mask.shape != ids.shape or mask.dtype != torch.long):
            raise ValueError("Sessions require one nonempty tokenized conversation.")
        if (not generation_config.use_cache or generation_config.do_sample or generation_config.num_beams != 1
                or type(generation_config.max_new_tokens) is not int or generation_config.max_new_tokens < 1):
            raise ValueError("Use greedy cached decoding and positive max_new_tokens.")
        if ids.shape[1] + generation_config.max_new_tokens > min(
            self.model.model.max_context, self.model.config.max_position_embeddings
        ):
            raise ValueError("Prompt plus generation budget exceeds the HTTPS context limit.")
        reuse = 0
        if self.cached_ids is not None:
            if self.cache.assert_consistent() != self.cached_ids.shape[1]:
                raise ValueError("Local history/cache mismatch.")
            common = min(self.cached_ids.shape[1], ids.shape[1])
            equal = (self.cached_ids[:, :common] == ids[:, :common]) & (self.cached_mask[:, :common] == mask[:, :common])
            mismatch = (~equal[0]).nonzero()
            reuse = min(int(mismatch[0].item()) if len(mismatch) else common, ids.shape[1] - 1)
            if reuse == 0:
                self.reset()
            elif reuse < self.cache.get_seq_length():
                self.cache.crop(reuse)
        schedule = []

        def observe(_model, _args, kwargs):
            schedule.append(kwargs["input_ids"].shape[1])

        hook = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            # Even a pre-generation CUDA error must release the temporary tracing hook.
            synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, generation_config=copy.deepcopy(generation_config),
                    past_key_values=self.cache, return_dict_in_generate=True, output_logits=trace,
                )
            synchronize()
            seconds = time.perf_counter() - start
            length = self.cache.assert_consistent()
            if length != output.sequences.shape[1] - 1:
                raise RuntimeError("Unexpected cache length after greedy generation.")
            self.cached_ids = output.sequences[:, :length].detach().clone()
            tail = torch.ones(1, output.sequences.shape[1] - ids.shape[1], dtype=mask.dtype, device=mask.device)
            self.cached_mask = torch.cat((mask, tail), dim=1)[:, :length].clone()
            return output, {
                "input_tokens": ids.shape[1], "reused_prefix_tokens": reuse,
                "prefill_tokens": ids.shape[1] - reuse, "cached_tokens": length,
                "query_tokens_per_forward": schedule,
                "generated_tokens_including_eos": output.sequences.shape[1] - ids.shape[1],
                "generation_seconds": seconds,
            }
        finally:
            hook.remove()
