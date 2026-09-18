"""Cache-enabled Qwen2/Qwen3-MoE adapters; stage 2 remains an independent control."""

import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
import transformers
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedsea.partition import Partition
from partition_cache import PartitionedCache

SUPPORTED_TRANSFORMERS = "4.56.1"


@dataclass(frozen=True)
class StepContext:
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple


class CachedBlockStage(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, context, cache):
        for layer in self.layers:
            hidden = layer(
                hidden, attention_mask=context.attention_mask,
                position_ids=context.position_ids, cache_position=context.cache_position,
                position_embeddings=context.position_embeddings,
                past_key_values=cache, use_cache=True,
            )
        return hidden


class CachedFrontStage(CachedBlockStage):
    def __init__(self, source, p):
        super().__init__(list(source.layers[:p]))
        self.embed_tokens = source.embed_tokens
        self.rotary_emb = source.rotary_emb


class CachedBackStage(CachedBlockStage):
    def __init__(self, source, lm_head, q):
        super().__init__(list(source.layers[-q:]))
        self.norm = source.norm
        self.lm_head = lm_head

    def forward(self, hidden, context, cache):
        return self.norm(super().forward(hidden, context, cache))


class CachedBackbone(nn.Module):
    def __init__(self, reference, partition):
        super().__init__()
        self.config = copy.deepcopy(reference.config)
        self.partition = partition
        self.cache_owner = object()
        source = reference.model
        self.stage_a = CachedFrontStage(source, partition.p)
        self.stage_b = CachedBlockStage(list(source.layers[partition.p:partition.p + partition.k]))
        self.stage_c = CachedBackStage(source, reference.lm_head, partition.q)
        self.output_type = (
            MoeModelOutputWithPast if self.config.model_type == "qwen3_moe"
            else BaseModelOutputWithPast
        )

    def get_input_embeddings(self):
        return self.stage_a.embed_tokens

    def new_cache(self):
        return PartitionedCache(self.config, self.partition, self.cache_owner)

    def validate_cache(self, cache):
        if not isinstance(cache, PartitionedCache) or cache.owner is not self.cache_owner:
            raise ValueError("Use a partitioned cache created by this exact split model.")
        return cache.assert_consistent()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                cache_position=None, **kwargs):
        if use_cache is False:
            raise ValueError("Stage 3 requires use_cache=True; stage 2 is the no-cache control.")
        if any(kwargs.get(key) for key in (
            "output_hidden_states", "output_attentions", "output_router_logits"
        )):
            raise ValueError("Full hidden/attention/router traces are unsupported; use boundary hooks.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        if input_ids is not None and (input_ids.ndim != 2 or input_ids.dtype != torch.long):
            raise ValueError("input_ids must be int64 [batch, new_tokens].")
        cache = past_key_values if past_key_values is not None else self.new_cache()
        past_length = self.validate_cache(cache)
        hidden = self.stage_a.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[1] < 1:
            raise ValueError("Embeddings must have nonempty [batch, new_tokens, hidden] shape.")
        batch, length, width = hidden.shape
        # hidden contains only NEW tokens; attention_mask must cover cached history as well.
        total = past_length + length
        if width != self.config.hidden_size or total > self.config.max_position_embeddings:
            raise ValueError("Input exceeds the model hidden-size/context contract.")
        if past_length:
            keys = cache.layers[0].keys
            if keys.shape[0] != batch or keys.device != hidden.device or keys.dtype != hidden.dtype:
                raise ValueError("Batch/device/dtype changed with a populated cache; reset first.")
        if attention_mask is None:
            attention_mask = torch.ones(batch, total, dtype=torch.long, device=hidden.device)
        if attention_mask.shape != (batch, total):
            raise ValueError("attention_mask must cover both cached history and new tokens.")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one.")
        if cache.attention_mask is not None and not torch.equal(
            attention_mask[:, :past_length], cache.attention_mask
        ):
            raise ValueError("Attention mask changed in the cached prefix; crop/reset first.")
        expected = torch.arange(past_length, total, device=hidden.device)
        if cache_position is None:
            cache_position = expected
        elif cache_position.dtype != torch.long or not torch.equal(cache_position, expected):
            raise ValueError("cache_position must continue the cache contiguously, without replay or gaps.")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must describe only the new tokens.")
        if torch.any(position_ids < 0) or torch.any(position_ids >= self.config.max_position_embeddings):
            raise ValueError("position_ids are outside the model context.")
        context = StepContext(
            create_causal_mask(
                config=self.config, input_embeds=hidden, attention_mask=attention_mask,
                cache_position=cache_position, past_key_values=cache, position_ids=position_ids,
            ),
            position_ids, cache_position, self.stage_a.rotary_emb(hidden, position_ids),
        )
        try:
            # Each segment appends to its own K/V, while preserving the original global layer IDs.
            hidden = self.stage_a(hidden, context, cache.a)
            hidden = self.stage_b(hidden, context, cache.b)
            hidden = self.stage_c(hidden, context, cache.c)
            if cache.assert_consistent() != total:
                raise RuntimeError("A/B/C did not consume the same number of new tokens.")
            cache.attention_mask = attention_mask.detach().clone()
        except BaseException:
            cache.valid = False
            raise
        return self.output_type(last_hidden_state=hidden, past_key_values=cache)


class CachedGenerationMixin:
    @property
    def lm_head(self):
        return self.model.stage_c.lm_head

    def new_cache(self):
        return self.model.new_cache()

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                      assistant_model, batch_size, max_cache_length):
        if (generation_config.use_cache is False
                or generation_config.get_generation_mode(assistant_model).value != "greedy_search"):
            raise ValueError("Stage 3 currently supports cache-enabled greedy generation only.")
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("Static/offloaded/quantized caches are not implemented in stage 3.")
        if generation_config.return_legacy_cache:
            raise ValueError("Legacy tuple caches would lose partition ownership.")
        cache = model_kwargs.get("past_key_values")
        if cache is None:
            model_kwargs["past_key_values"] = self.new_cache()
        else:
            self.model.validate_cache(cache)


def initialize_cached(wrapper, reference, partition, model_class, pretrained_class):
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(f"Stage 3 requires transformers=={SUPPORTED_TRANSFORMERS}.")
    if not isinstance(reference, model_class) or reference.training:
        raise ValueError(f"This adapter requires an eval-mode {model_class.__name__}.")
    partition.validate(reference.config.num_hidden_layers)
    config = copy.deepcopy(reference.config)
    if config._attn_implementation != "eager":
        raise ValueError("The verified stage-3 path requires eager attention.")
    if getattr(config, "sliding_window", None) is not None or any(
        value != "full_attention" for value in getattr(config, "layer_types", [])
    ):
        raise ValueError("Sliding/hybrid attention is not supported by the stage-3 session policy.")
    config.use_cache = True
    if hasattr(config, "output_router_logits"):
        config.output_router_logits = False
    # Initialize HF metadata only, retaining the original decoder and weight objects.
    pretrained_class.__init__(wrapper, config)
    wrapper.model = CachedBackbone(reference, partition)
    wrapper.vocab_size = reference.vocab_size
    wrapper.partition = partition
    wrapper.generation_config = copy.deepcopy(reference.generation_config)
    wrapper.generation_config.use_cache = True
    for key in ("router_aux_loss_coef", "num_experts", "num_experts_per_tok"):
        if hasattr(reference, key):
            setattr(wrapper, key, getattr(reference, key))
    wrapper.eval()


class CachedQwen2ForCausalLM(CachedGenerationMixin, Qwen2ForCausalLM):
    def __init__(self, reference, partition):
        initialize_cached(self, reference, partition, Qwen2ForCausalLM, Qwen2PreTrainedModel)


class CachedQwen3MoeForCausalLM(CachedGenerationMixin, Qwen3MoeForCausalLM):
    def __init__(self, reference, partition):
        initialize_cached(self, reference, partition, Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel)


CACHED_ADAPTERS = {"qwen2": CachedQwen2ForCausalLM, "qwen3_moe": CachedQwen3MoeForCausalLM}


def build_cached_model(reference, partition, adapter=None):
    key = adapter or reference.config.model_type
    if key != reference.config.model_type or key not in CACHED_ADAPTERS:
        raise ValueError("No matching cache-enabled adapter for this model architecture.")
    return CACHED_ADAPTERS[key](reference, partition)
