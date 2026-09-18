"""Model identities, loading gates, and inspectable profiles. No GPU work here."""

import copy
import re
from dataclasses import dataclass

from .project import ROOT, project_path, read_json

IDENTITY_FIELDS = (
    "repo_id", "revision", "local_dir", "model_type",
    "num_hidden_layers", "hidden_size", "weights_sha256",
)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    settings: dict

    @property
    def spec(self):
        return {key: self.settings[key] for key in IDENTITY_FIELDS if key in self.settings}

    @property
    def template_kwargs(self):
        return copy.deepcopy(self.settings.get("chat_template_kwargs", {}))

    def require_enabled(self):
        if not self.settings["enabled"]:
            raise ValueError(
                f"Model '{self.name}' is reserved, not deployed/enabled. "
                "Pin its revision and plan sufficient hardware first. See docs/models.md. "
                "Use --describe to inspect without loading weights."
            )
        revision = self.settings.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("An enabled model must pin a 40-character lowercase commit revision.")

    def describe(self):
        return {"name": self.name, **copy.deepcopy(self.settings)}


class ModelRegistry:
    def __init__(self, path=None):
        data = read_json(path or ROOT / "configs/model.json")
        self.default = data["default_model"]
        self.models = data["models"]
        if self.default not in self.models:
            raise ValueError("default_model is not present in the model registry.")
        destinations = set()
        for name, settings in self.models.items():
            if type(settings.get("enabled")) is not bool:
                raise ValueError(f"{name}: enabled must be a boolean.")
            for key in ("num_hidden_layers", "hidden_size", "parameter_count"):
                if type(settings.get(key)) is not int or settings[key] <= 0:
                    raise ValueError(f"{name}: {key} must be a positive integer.")
            for key in ("repo_id", "model_type", "adapter"):
                if not isinstance(settings.get(key), str) or not settings[key]:
                    raise ValueError(f"{name}: {key} must be a nonempty string.")
            directory = project_path(settings["local_dir"])
            if not directory.is_relative_to(ROOT / "models") or directory == ROOT / "models":
                raise ValueError("Each model must have its own directory under models/.")
            if directory in destinations:
                raise ValueError("Two model profiles cannot share one model directory.")
            destinations.add(directory)
            kwargs = settings.get("chat_template_kwargs", {})
            if not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"}:
                raise ValueError("Only enable_thinking is currently supported as a template override.")
            if "enable_thinking" in kwargs and type(kwargs["enable_thinking"]) is not bool:
                raise ValueError("enable_thinking must be a boolean.")
            if settings["enabled"]:
                ModelProfile(name, settings).require_enabled()

    def get(self, name=None):
        name = name or self.default
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(self.models)}")
        return ModelProfile(name, copy.deepcopy(self.models[name]))

    def list_models(self):
        for name in self.models:
            profile = self.get(name)
            status = "enabled" if profile.settings["enabled"] else "reserved"
            print(f"{name} | {status} | {profile.settings['num_hidden_layers']} layers | "
                  f"{profile.settings['adapter']} | {profile.settings['repo_id']}")
