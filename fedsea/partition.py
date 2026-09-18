"""Contiguous A/B/C partition selection; middle blocks always follow model depth."""

from dataclasses import dataclass

from .project import ROOT, read_json


@dataclass(frozen=True)
class Partition:
    p: int
    k: int
    q: int

    def validate(self, total):
        if type(total) is not int or total < 3:
            raise ValueError("The model needs at least three decoder layers.")
        if any(type(n) is not int or n <= 0 for n in (self.p, self.k, self.q)):
            raise ValueError("p, k, q must all be positive integers.")
        if self.p + self.k + self.q != total:
            raise ValueError(f"p + k + q must equal the model's {total} decoder layers.")

    def describe(self):
        return {
            "A": {"layers_1based": [1, self.p], "embedding": True},
            "B": {"layers_1based": [self.p + 1, self.p + self.k]},
            "C": {"layers_1based": [self.p + self.k + 1, self.p + self.k + self.q],
                  "final_norm": True, "lm_head": True},
        }


def resolve_partition(total, selection=None, *, preset=None, p=None, q=None, presets=None):
    # Resolve in order: preset -> stage overrides -> CLI p/q; only then derive k.
    presets = presets or read_json(ROOT / "configs/splits.json")
    selection = selection or {"preset": presets["default_preset"]}
    if set(selection) - {"preset", "p", "q"}:
        raise ValueError("Split selection supports preset, p, q only; k is derived.")
    selected = preset or selection.get("preset")
    if selected:
        if selected not in presets["presets"]:
            raise ValueError(f"Unknown split preset '{selected}'.")
        values = dict(presets["presets"][selected])
    else:
        values = {}
    if preset is None:
        values.update({key: selection[key] for key in ("p", "q") if key in selection})
    if p is not None:
        values["p"] = p
    if q is not None:
        values["q"] = q
    if set(values) != {"p", "q"} or any(type(v) is not int for v in values.values()):
        raise ValueError("A split must supply integer p and q.")
    # Embedding, final norm and LM head are not counted as decoder layers.
    partition = Partition(values["p"], total - values["p"] - values["q"], values["q"])
    partition.validate(total)
    return partition
