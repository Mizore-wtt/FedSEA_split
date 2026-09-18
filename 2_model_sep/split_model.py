"""Registered Qwen2 / Qwen3-MoE A/B/C adapters, without a second weight copy."""

import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import transformers
from torch import nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedsea.partition import Partition

SUPPORTED_TRANSFORMERS = "4.56.1"


@dataclass(frozen=True)
class ForwardContext:
    masks: dict
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple


class BlockStage(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, context):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=context.masks[getattr(layer, "attention_type", "full_attention")],
                position_ids=context.position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=context.cache_position,
                position_embeddings=context.position_embeddings,
            )
        return hidden_states


class FrontStage(BlockStage):
    """Token IDs [B,T] -> embeddings [B,T,H] -> the first p decoder layers."""
    def __init__(self, source, p):
        super().__init__(list(source.layers[:p]))
        self.config = source.config
        self.embed_tokens = source.embed_tokens
        self.rotary_emb = source.rotary_emb
        self.has_sliding_layers = getattr(source, "has_sliding_layers", False)

    def forward(self, input_ids, inputs_embeds, attention_mask, position_ids, cache_position):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        if input_ids is not None and input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[1] == 0:
            raise ValueError("Embeddings must be [batch, nonempty sequence, hidden].")
        batch, length, width = hidden.shape
        if width != self.config.hidden_size or length > self.config.max_position_embeddings:
            raise ValueError("Input exceeds the model's hidden-size/context contract.")
        if attention_mask is not None and attention_mask.shape != (batch, length):
            raise ValueError("Stage 2 expects a 2D attention mask matching the full input.")
        # No cache: every generation step reevaluates the whole sequence from position zero.
        expected_positions = torch.arange(length, device=hidden.device)
        if cache_position is None:
            cache_position = expected_positions
        elif not torch.equal(cache_position, expected_positions):
            raise ValueError("Without KV cache, cache_position must cover the full sequence from zero.")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence].")
        mask_kwargs = {
            "config": self.config,
            "input_embeds": hidden,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if self.config.model_type == "qwen3_moe":
            mask_function = (
                create_sliding_window_causal_mask
                if self.config.sliding_window is not None else create_causal_mask
            )
            masks = {"full_attention": mask_function(**mask_kwargs)}
        else:
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.config.model_type == "qwen2" and self.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        context = ForwardContext(
            masks, position_ids, cache_position, self.rotary_emb(hidden, position_ids)
        )
        return super().forward(hidden, context), context


class BackStage(BlockStage):
    """Last q decoder layers plus final norm; the native LM wrapper applies lm_head."""
    def __init__(self, source, lm_head, q):
        super().__init__(list(source.layers[-q:]))
        self.norm = source.norm
        self.lm_head = lm_head

    def forward(self, hidden_states, context):
        return self.norm(super().forward(hidden_states, context))


class SplitBackbone(nn.Module):
    def __init__(self, source, lm_head, partition):
        super().__init__()
        self.output_type = (
            MoeModelOutputWithPast if source.config.model_type == "qwen3_moe"
            else BaseModelOutputWithPast
        )
        self.stage_a = FrontStage(source, partition.p)
        self.stage_b = BlockStage(list(source.layers[partition.p:partition.p + partition.k]))
        self.stage_c = BackStage(source, lm_head, partition.q)

    def get_input_embeddings(self):
        return self.stage_a.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False,
                cache_position=None, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError("Stage 2 does not support KV cache. Use use_cache=False.")
        if any(kwargs.get(key) for key in ("output_hidden_states", "output_attentions", "output_router_logits")):
            raise ValueError("Use boundary hooks for stage-2 inspection; full/router traces are unsupported.")
        # A/B/C share one positional context. B must not restart RoPE positions from zero.
        hidden, context = self.stage_a(
            input_ids, inputs_embeds, attention_mask, position_ids, cache_position
        )
        hidden = self.stage_b(hidden, context)
        hidden = self.stage_c(hidden, context)
        return self.output_type(last_hidden_state=hidden, past_key_values=None)


def initialize_split(wrapper, reference, partition, model_class, pretrained_class):
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(f"Stage 2 requires transformers=={SUPPORTED_TRANSFORMERS}.")
    if not isinstance(reference, model_class):
        raise ValueError(f"This adapter requires {model_class.__name__}.")
    partition.validate(reference.config.num_hidden_layers)
    if reference.training:
        raise ValueError("The reference model must already be in eval mode.")
    config = copy.deepcopy(reference.config)
    config.use_cache = False
    if hasattr(config, "output_router_logits"):
        config.output_router_logits = False
    # Initialize HF metadata, not a second set of randomly initialized weights.
    pretrained_class.__init__(wrapper, config)
    wrapper.model = SplitBackbone(reference.model, reference.lm_head, partition)
    wrapper.vocab_size = reference.vocab_size
    wrapper.partition = partition
    wrapper.generation_config = copy.deepcopy(reference.generation_config)
    wrapper.generation_config.use_cache = False
    for attribute in ("router_aux_loss_coef", "num_experts", "num_experts_per_tok"):
        if hasattr(reference, attribute):
            setattr(wrapper, attribute, getattr(reference, attribute))
    wrapper.eval()


class SplitQwen2ForCausalLM(Qwen2ForCausalLM):
    """Reuse HF's LM output and generate(), replacing only the decoder traversal."""

    def __init__(self, reference, partition):
        initialize_split(self, reference, partition, Qwen2ForCausalLM, Qwen2PreTrainedModel)

    @property
    def lm_head(self):
        return self.model.stage_c.lm_head


class SplitQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Keep complete MoE decoder blocks, including the original routing and experts."""

    def __init__(self, reference, partition):
        initialize_split(self, reference, partition, Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel)

    @property
    def lm_head(self):
        return self.model.stage_c.lm_head


SPLIT_ADAPTERS = {
    "qwen2": SplitQwen2ForCausalLM,
    "qwen3_moe": SplitQwen3MoeForCausalLM,
}


def build_split_model(reference, partition, adapter=None):
    # Model selection and layer counts are independent: adapters own architecture differences.
    model_type = reference.config.model_type
    adapter = adapter or model_type
    if adapter != model_type:
        raise ValueError("Selected split adapter does not match the model's architecture.")
    if adapter not in SPLIT_ADAPTERS:
        raise ValueError(f"No split adapter registered for '{adapter}'. See docs/models.md.")
    return SPLIT_ADAPTERS[adapter](reference, partition)
