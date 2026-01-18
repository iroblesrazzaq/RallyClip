from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from training.features.v1 import FeatureSetV1


@dataclass
class FeatureSetInfo:
    name: str
    builder: Type


class FeatureRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, FeatureSetInfo] = {}
        self.register("v1", FeatureSetV1)

    def register(self, name: str, builder: Type) -> None:
        self._registry[name] = FeatureSetInfo(name=name, builder=builder)

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"Unknown feature set: {name}")
        return self._registry[name].builder
