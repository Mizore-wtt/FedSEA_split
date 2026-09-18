"""Owned dynamic caches, native decoder execution, and fail-closed RPC sessions."""

import copy
import secrets
import threading
import time
import os

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel

from .protocol import validate_step


class RangeCache(Cache):
    def __init__(self, start, count):
        super().__init__(layers=[DynamicLayer() for _ in range(count)])
        self.start, self.stop = start, start + count

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        if not self.start <= layer_idx < self.stop:
            raise ValueError("Layer wrote outside its owned cache.")
        return super().update(keys, values, layer_idx - self.start, cache_kwargs)

    def length(self):
        lengths = [layer.get_seq_length() for layer in self.layers]
        if len(set(lengths)) != 1:
            raise RuntimeError("Owned cache layers have inconsistent lengths.")
        return lengths[0]


def context(weights, hidden, tensors, cache):
    positions, cache_pos = tensors["position_ids"], tensors["cache_position"]
    return {
        "attention_mask": create_causal_mask(
            config=weights.config, input_embeds=hidden, attention_mask=tensors["attention_mask"],
            cache_position=cache_pos, past_key_values=cache, position_ids=positions,
        ),
        "position_ids": positions, "cache_position": cache_pos,
        "position_embeddings": weights.rotary_emb(hidden, positions),
    }


class ServerState:
    def __init__(self, weights, identity, net, clock=time.monotonic, instance_id=None):
        if weights.role != "server":
            raise ValueError("ServerState must own only B weights.")
        self.weights, self.identity, self.net = weights, identity, net
        self.clock, self.sessions, self.lock = clock, {}, threading.Lock()
        self.instance_id = instance_id

    def expire(self):
        with self.lock:
            now = self.clock()
            for sid in list(self.sessions):
                if now - self.sessions[sid]["last"] >= self.net["session_ttl_seconds"]:
                    del self.sessions[sid]

    def dispatch(self, path, meta, tensors):
        self.expire()
        # Inference mode is thread-local; serialize GPU work and KV mutations under one lock.
        with self.lock, torch.inference_mode():
            if path == "/v1/health":
                if meta or tensors:
                    raise ValueError("Health takes no data.")
                return {
                    "identity": self.identity, "pid": os.getpid(),
                    "instance_id": self.instance_id,
                    "active_sessions": len(self.sessions),
                    "max_context_tokens": self.net["max_context_tokens"],
                    "max_body_bytes": self.net["max_body_bytes"],
                    "weights": self.weights.describe(),
                }, {}
            if path == "/v1/session":
                if set(meta) != {"identity"} or tensors or meta["identity"] != self.identity:
                    raise ValueError("Client/server model, precision, revision, or split differs.")
                if len(self.sessions) >= self.net["max_sessions"]:
                    raise ValueError("Session capacity reached; close unused sessions or wait for TTL.")
                sid = secrets.token_urlsafe(32)
                part = self.weights.partition
                self.sessions[sid] = {
                    "cache": RangeCache(part.p, part.k), "mask": None, "seq": 0,
                    "last": self.clock(),
                }
                return {"session": sid, "seq": 0, "length": 0}, {}
            sid = meta.get("session")
            if not isinstance(sid, str) or len(sid) != 43:
                raise ValueError("Invalid session identifier.")
            if path == "/v1/close":
                if set(meta) != {"session"} or tensors:
                    raise ValueError("Close takes only a session identifier.")
                self.sessions.pop(sid, None)
                return {"closed": True}, {}
            if sid not in self.sessions:
                raise ValueError("Session expired, was closed, or does not exist; reset locally.")
            session = self.sessions[sid]
            try:
                fields = {"session", "seq", "past"}
                if path == "/v1/crop":
                    fields.add("keep")
                if (set(meta) != fields or type(meta["seq"]) is not int
                        or type(meta["past"]) is not int or meta["seq"] != session["seq"]
                        or meta["past"] != session["cache"].length()):
                    raise ValueError("Request order/cache length mismatch; session invalidated.")
                if path == "/v1/forward":
                    total = validate_step(
                        tensors, self.weights.config, self.weights.dtype, meta["past"],
                        self.net["max_context_tokens"], session["mask"],
                    )
                    device_tensors = {k: v.to(self.weights.device) for k, v in tensors.items()}
                    hidden = device_tensors["hidden"]
                    result = self.weights.run(
                        hidden, context(self.weights, hidden, device_tensors, session["cache"]),
                        session["cache"],
                    )
                    if session["cache"].length() != total or not torch.isfinite(result).all():
                        raise RuntimeError("Server produced an invalid result/cache.")
                    # Copy before committing metadata; any GPU/copy error invalidates the session.
                    output = {"hidden": result.detach().to("cpu").contiguous()}
                    session["mask"] = tensors["attention_mask"].clone()
                elif path == "/v1/crop":
                    total = meta["keep"]
                    if tensors or type(total) is not int or not 0 <= total <= meta["past"]:
                        raise ValueError("Invalid cache crop.")
                    if total == 0:
                        part = self.weights.partition
                        session["cache"] = RangeCache(part.p, part.k)
                        session["mask"] = None
                    else:
                        session["cache"].crop(total)
                        session["mask"] = session["mask"][:, :total].clone()
                    output = {}
                else:
                    raise ValueError("Unknown operation.")
                session["seq"] += 1
                session["last"] = self.clock()
                return {"session": sid, "seq": session["seq"], "length": total}, output
            except BaseException:
                self.sessions.pop(sid, None)
                raise


class RemoteCache(Cache):
    def __init__(self, config, part, rpc, identity, owner):
        self.config, self.part, self.rpc, self.identity, self.owner = config, part, rpc, identity, owner
        self.sid = None
        self.reset()

    def _empty(self):
        self.a = RangeCache(0, self.part.p)
        self.c = RangeCache(self.part.p + self.part.k, self.part.q)
        super().__init__(layers=self.a.layers + self.c.layers)
        self.remote_length, self.seq, self.attention_mask = 0, 0, None

    def close(self):
        sid, self.sid = self.sid, None
        self._empty()
        self.valid = False
        if sid is not None:
            try:
                self.rpc.call("/v1/close", {"session": sid})
            except Exception:
                # Unreachable server sessions expire by TTL. Never replay a forward.
                pass

    def reset(self):
        if self.sid is not None:
            self.close()
        self._empty()
        self.valid = True

    def update(self, *args, **kwargs):
        raise ValueError("Write through the owning A or C cache.")

    def assert_consistent(self):
        if not self.valid:
            raise ValueError("Cache invalidated; reset the session.")
        if self.a.length() != self.c.length() or self.a.length() != self.remote_length:
            raise ValueError("A/B/C cache lengths differ.")
        return self.remote_length

    def begin(self):
        if self.sid is None:
            reply, tensors = self.rpc.call("/v1/session", {"identity": self.identity})
            self.sid = reply.get("session")
            if (set(reply) != {"session", "seq", "length"} or tensors
                    or not isinstance(self.sid, str) or len(self.sid) != 43
                    or type(reply["seq"]) is not int or type(reply["length"]) is not int
                    or reply["seq"] != 0 or reply["length"] != 0):
                raise ValueError("Invalid create-session response.")

    def operation(self, path, expected_length, tensors=None, **extra):
        # The server acknowledges both operation order and KV length before local state advances.
        self.begin()
        reply, output = self.rpc.call(path, {
            "session": self.sid, "seq": self.seq, "past": self.remote_length, **extra,
        }, tensors)
        if (type(reply.get("seq")) is not int or type(reply.get("length")) is not int
                or reply != {"session": self.sid, "seq": self.seq + 1, "length": expected_length}):
            raise ValueError("Unexpected server operation response.")
        self.seq += 1
        self.remote_length = expected_length
        return output

    def crop(self, length):
        past = self.assert_consistent()
        if type(length) is not int or not 0 <= length <= past:
            raise ValueError("Crop must retain an existing prefix.")
        if length == past:
            return
        if length == 0:
            self.reset()
            return
        try:
            if self.operation("/v1/crop", length, keep=length):
                raise ValueError("Crop returned unexpected tensors.")
            self.a.crop(length)
            self.c.crop(length)
            self.attention_mask = self.attention_mask[:, :length].clone()
            self.assert_consistent()
        except BaseException:
            self.close()
            raise


class RemoteBackbone(nn.Module):
    def __init__(self, weights, rpc, identity, max_context):
        super().__init__()
        if weights.role != "client":
            raise ValueError("RemoteBackbone must own only A/C weights.")
        self.weights, self.config, self.partition = weights, weights.config, weights.partition
        self.rpc, self.identity, self.max_context = rpc, identity, max_context
        self.owner = object()

    def get_input_embeddings(self):
        return self.weights.embed_tokens

    def new_cache(self):
        return RemoteCache(self.config, self.partition, self.rpc, self.identity, self.owner)

    def validate_cache(self, cache):
        if not isinstance(cache, RemoteCache) or cache.owner is not self.owner:
            raise ValueError("Use a cache created by this exact HTTPS client model.")
        return cache.assert_consistent()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
        if use_cache is False or inputs_embeds is not None:
            raise ValueError("HTTPS supports cached input_ids inference only.")
        if any(kwargs.get(key) for key in ("output_attentions", "output_hidden_states", "output_router_logits")):
            raise ValueError("Full internal traces/router outputs are unsupported over HTTPS.")
        if input_ids is None or input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be int64 [1, new_tokens].")
        cache = past_key_values if past_key_values is not None else self.new_cache()
        past = self.validate_cache(cache)
        try:
            hidden = self.weights.embed_tokens(input_ids)
            length = hidden.shape[1]
            if cache_position is None:
                cache_position = torch.arange(past, past + length, device=hidden.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            if attention_mask is None:
                attention_mask = torch.ones(1, past + length, device=hidden.device, dtype=torch.long)
            tensors = {
                "hidden": hidden, "attention_mask": attention_mask,
                "position_ids": position_ids, "cache_position": cache_position,
            }
            total = validate_step(tensors, self.config, self.weights.dtype, past,
                                  self.max_context, cache.attention_mask)
            cache.begin()
            ctx = context(self.weights, hidden, tensors, cache)
            hidden = self.weights.run(hidden, ctx, cache.a, range(self.partition.p))
            # Token IDs remain here. Only A's hidden states, mask and positions are uploaded.
            tensors["hidden"] = hidden
            result = cache.operation("/v1/forward", total, tensors)
            if (set(result) != {"hidden"} or result["hidden"].shape != hidden.shape
                    or result["hidden"].dtype != hidden.dtype or not torch.isfinite(result["hidden"]).all()):
                raise ValueError("Invalid B output.")
            hidden = result["hidden"].to(hidden.device)
            # Resume at C using the SAME positions as A and B; do not apply embedding a second time.
            hidden = self.weights.run(
                hidden, ctx, cache.c, range(self.partition.p + self.partition.k, self.config.num_hidden_layers)
            )
            hidden = self.weights.norm(hidden)
            if not torch.isfinite(hidden).all() or cache.assert_consistent() != total:
                raise RuntimeError("Invalid C output/cache.")
            cache.attention_mask = attention_mask.detach().clone()
        except BaseException:
            cache.close()
            raise
        output_type = MoeModelOutputWithPast if self.config.model_type == "qwen3_moe" else BaseModelOutputWithPast
        return output_type(last_hidden_state=hidden, past_key_values=cache)


class RemoteGenerationMixin:
    @property
    def lm_head(self):
        return self.model.weights.lm_head

    def new_cache(self):
        return self.model.new_cache()

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                      assistant_model, batch_size, max_cache_length):
        if (generation_config.use_cache is False or batch_size != 1
                or generation_config.get_generation_mode(assistant_model).value != "greedy_search"
                or generation_config.cache_implementation not in (None, "dynamic")
                or generation_config.return_legacy_cache):
            raise ValueError("HTTPS currently supports batch-1 greedy generation with dynamic caches.")
        if model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = self.new_cache()
        self.model.validate_cache(model_kwargs["past_key_values"])


def initialize(wrapper, weights, rpc, identity, max_context, pretrained):
    config = copy.deepcopy(weights.config)
    pretrained.__init__(wrapper, config)
    wrapper.model = RemoteBackbone(weights, rpc, identity, max_context)
    wrapper.vocab_size, wrapper.partition = config.vocab_size, weights.partition
    for name in ("router_aux_loss_coef", "num_experts", "num_experts_per_tok"):
        if hasattr(config, name):
            setattr(wrapper, name, getattr(config, name))
    wrapper.eval().requires_grad_(False)


class RemoteQwen2(RemoteGenerationMixin, Qwen2ForCausalLM):
    def __init__(self, weights, rpc, identity, max_context):
        initialize(self, weights, rpc, identity, max_context, Qwen2PreTrainedModel)


class RemoteQwen3Moe(RemoteGenerationMixin, Qwen3MoeForCausalLM):
    def __init__(self, weights, rpc, identity, max_context):
        initialize(self, weights, rpc, identity, max_context, Qwen3MoePreTrainedModel)


REMOTE_ADAPTERS = {"qwen2": RemoteQwen2, "qwen3_moe": RemoteQwen3Moe}


def build_client(weights, rpc, identity, max_context):
    return REMOTE_ADAPTERS[weights.config.model_type](weights, rpc, identity, max_context)
