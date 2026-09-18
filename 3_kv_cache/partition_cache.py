"""Separate A/B/C ownership of native Transformers dynamic cache layers."""

import copy

from transformers.cache_utils import Cache, DynamicCache


class LayerRangeCache(Cache):
    """Own local cache layers, accepting the original global index only on update."""

    def __init__(self, layers, start):
        super().__init__(layers=layers)
        self.start = start
        self.stop = start + len(layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Decoder IDs stay global; e.g. layer 2 writes slot 0 in B when p=2.
        if not self.start <= layer_idx < self.stop:
            raise ValueError("A decoder layer tried to write outside its cache partition.")
        return super().update(key_states, value_states, layer_idx - self.start, cache_kwargs)


class PartitionedCache(Cache):
    """HF-compatible global read view; decoder writes must go through A/B/C."""

    def __init__(self, config, partition, owner):
        partition.validate(config.num_hidden_layers)
        self.model_config = copy.deepcopy(config)
        self.partition = partition
        self.owner = owner
        self.reset()

    def reset(self):
        # DynamicLayer.reset() zeroes storage without shortening it in HF 4.56.1.
        # New native layers actually release this session's history and tensors.
        native = DynamicCache(config=self.model_config)
        if any(native.is_sliding):
            raise ValueError("Stage 3 supports full-attention dynamic caches, not sliding-window caches.")
        p, k = self.partition.p, self.partition.k
        self.a = LayerRangeCache(native.layers[:p], 0)
        self.b = LayerRangeCache(native.layers[p:p + k], p)
        self.c = LayerRangeCache(native.layers[p + k:], p + k)
        # The global view references the same cache layers, not a fourth copy of the KV tensors.
        super().__init__(layers=self.a.layers + self.b.layers + self.c.layers)
        self.valid = True
        self.attention_mask = None

    def update(self, *args, **kwargs):
        raise ValueError("Decoder writes must use the owning segment cache, not the global view.")

    def assert_consistent(self):
        if not self.valid:
            raise ValueError("Cache was invalidated by a failed forward; reset the session.")
        lengths = [layer.get_seq_length() for layer in self.layers]
        if len(set(lengths)) != 1:
            self.valid = False
            raise ValueError("Cache layers disagree on history length; reset the session.")
        return lengths[0]

    def crop(self, max_length):
        old_length = self.assert_consistent()
        if type(max_length) is not int or not 0 <= max_length <= old_length:
            raise ValueError("Cache crop must keep a nonnegative prefix of the existing history.")
        super().crop(max_length)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask[:, :max_length].clone()

    def describe(self):
        length = self.assert_consistent()
        result = {}
        for name, segment in (("A", self.a), ("B", self.b), ("C", self.c)):
            size = sum(
                tensor.numel() * tensor.element_size()
                for layer in segment.layers for tensor in (layer.keys, layer.values)
                if tensor is not None
            )
            result[name] = {
                "layers_1based": [segment.start + 1, segment.stop],
                "layer_count": len(segment.layers), "cached_tokens": length, "kv_bytes": size,
            }
        return result
