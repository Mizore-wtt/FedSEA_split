"""Bounded JSON metadata plus safetensors bytes. Never pickle or token IDs."""

import json
import struct

import torch
from safetensors.torch import load, save

CONTENT_TYPE = "application/vnd.fedsea.safetensors"
MAX_METADATA = 16384
MAX_TENSORS = 4


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def json_object(data):
    value = json.loads(data, object_pairs_hook=unique_object,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def pack(metadata, tensors=None, limit=16 * 1024**2):
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(header) > MAX_METADATA:
        raise ValueError("Metadata is too large.")
    tensors = tensors or {}
    if len(tensors) > MAX_TENSORS:
        raise ValueError("Too many tensors.")
    # position_ids may be a view of cache_position; safetensors disallows shared storage.
    binary = save({k: v.detach().to("cpu").contiguous().clone() for k, v in tensors.items()}) if tensors else b""
    result = struct.pack("!I", len(header)) + header + binary
    if len(result) > limit:
        raise ValueError("Message exceeds max_body_bytes.")
    return result


def unpack(body, limit=16 * 1024**2):
    if not 4 <= len(body) <= limit:
        raise ValueError("Invalid body length.")
    size = struct.unpack("!I", body[:4])[0]
    if not 2 <= size <= MAX_METADATA or 4 + size > len(body):
        raise ValueError("Invalid metadata length.")
    metadata = json_object(body[4:4 + size])
    binary = body[4 + size:]
    if not binary:
        return metadata, {}
    if len(binary) < 8:
        raise ValueError("Truncated safetensors.")
    header_size = struct.unpack("<Q", binary[:8])[0]
    if not 2 <= header_size <= MAX_METADATA or 8 + header_size > len(binary):
        raise ValueError("Invalid safetensors header length.")
    header = json_object(binary[8:8 + header_size])
    # Validate advertised sizes before asking safetensors to materialize any network tensors.
    if "__metadata__" in header or not 1 <= len(header) <= MAX_TENSORS:
        raise ValueError("Unexpected tensor metadata/count.")
    widths = {"F16": 2, "BF16": 2, "F32": 4, "I64": 8}
    for item in header.values():
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Malformed tensor descriptor.")
        shape = item["shape"]
        if (item["dtype"] not in widths or not isinstance(shape, list)
                or not 1 <= len(shape) <= 3):
            raise ValueError("Unsupported tensor dtype/rank.")
        nbytes = widths[item["dtype"]]
        for dimension in shape:
            if type(dimension) is not int or not 1 <= dimension <= limit:
                raise ValueError("Invalid tensor dimension.")
            nbytes *= dimension
        if nbytes > limit:
            raise ValueError("Tensor allocation exceeds the message budget.")
    return metadata, load(binary)


def validate_step(tensors, config, dtype, past, max_context, old_mask=None):
    if set(tensors) != {"hidden", "attention_mask", "position_ids", "cache_position"}:
        raise ValueError("Only hidden states, masks, and positions may cross the boundary.")
    hidden, mask = tensors["hidden"], tensors["attention_mask"]
    positions, cache_pos = tensors["position_ids"], tensors["cache_position"]
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != config.hidden_size
            or hidden.dtype != dtype or hidden.shape[1] < 1 or not torch.isfinite(hidden).all()):
        raise ValueError("Invalid/nonfinite hidden states; only batch size 1 is supported.")
    length = hidden.shape[1]
    total = past + length
    if total > min(max_context, config.max_position_embeddings):
        raise ValueError("HTTPS context limit exceeded.")
    if (mask.dtype != torch.long or mask.shape != (1, total)
            or not torch.all((mask == 0) | (mask == 1))):
        raise ValueError("Invalid full-history attention mask.")
    if old_mask is not None and not torch.equal(mask[:, :past], old_mask):
        raise ValueError("Cached attention-mask prefix changed.")
    expected = torch.arange(past, total, device=cache_pos.device)
    if cache_pos.dtype != torch.long or cache_pos.shape != (length,) or not torch.equal(cache_pos, expected):
        raise ValueError("Cache positions must continue without replay or gaps.")
    if (positions.dtype != torch.long or positions.shape != (1, length)
            or torch.any(positions < 0) or torch.any(positions >= config.max_position_embeddings)):
        raise ValueError("Invalid RoPE positions.")
    return total
