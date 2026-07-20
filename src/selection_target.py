from __future__ import annotations

import math
from typing import Any, Mapping


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def selection_target_per_subset(cfg: Mapping[str, Any]) -> int:
    subset_size = int(_get(cfg, "data.subset_size", 100000) or 0)
    if subset_size <= 0:
        raise ValueError("data.subset_size must be > 0")
    selection_ratio = float(_get(cfg, "data.selection_ratio", 0.01) or 0.0)
    if selection_ratio <= 0:
        raise ValueError("data.selection_ratio must be > 0")
    return max(1, math.ceil(subset_size * selection_ratio))
