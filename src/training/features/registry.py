from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from training.features.v1 import FeatureSetV1
from training.features.v2 import FeatureSetV2, FeatureSetV2NearFar


@dataclass
class FeatureSetInfo:
    name: str
    builder: Type


class FeatureRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, FeatureSetInfo] = {}
        self.register("v1", FeatureSetV1)
        # v2: v1 minus the exactly-redundant magnitude groups (181 -> 145 per
        # slot). "v2" carries the far-crop slots (4 x 145 = 580); "v2_nearfar"
        # is the same layout without them (2 x 145 = 290) and is the baseline
        # that isolates what the crop actually contributes.
        self.register("v2", FeatureSetV2)
        self.register("v2_nearfar", FeatureSetV2NearFar)

    def register(self, name: str, builder: Type) -> None:
        self._registry[name] = FeatureSetInfo(name=name, builder=builder)

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"Unknown feature set: {name}")
        return self._registry[name].builder
