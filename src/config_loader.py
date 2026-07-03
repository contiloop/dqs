from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"cannot interpret as bool: {value!r}")


def _register_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "onoff",
        lambda value: "on" if _as_bool(value) else "off",
        replace=True,
    )


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    return loaded


def _default_groups(defaults: Any) -> set[str]:
    groups: set[str] = set()
    for item in defaults:
        if isinstance(item, DictConfig):
            groups.update(str(key) for key in item.keys())
    return groups


def _split_overrides(
    defaults: Any,
    overrides: list[str] | None,
) -> tuple[dict[str, str], list[str]]:
    groups = _default_groups(defaults)
    group_overrides: dict[str, str] = {}
    dot_overrides: list[str] = []
    for override in overrides or []:
        if "=" not in override:
            dot_overrides.append(override)
            continue
        key, value = override.split("=", 1)
        if key in groups and "." not in key:
            group_overrides[key] = value
        else:
            dot_overrides.append(override)
    return group_overrides, dot_overrides


def compose_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    _register_resolvers()
    root_path = Path(config_path)
    config_dir = root_path.parent
    root = _load_yaml(root_path)
    defaults = root.get("defaults")
    if defaults is None:
        raise ValueError(f"missing defaults list: {root_path}")

    group_overrides, dot_overrides = _split_overrides(defaults, overrides)
    merged = OmegaConf.create({})
    for item in defaults:
        if item == "_self_":
            continue
        if isinstance(item, str):
            group_path = config_dir / f"{item}.yaml"
        elif isinstance(item, DictConfig):
            entries = list(item.items())
            if len(entries) != 1:
                raise ValueError(f"invalid defaults entry: {item}")
            group, name = entries[0]
            name = group_overrides.get(str(group), str(name))
            group_path = config_dir / str(group) / f"{name}.yaml"
        else:
            raise ValueError(f"invalid defaults entry: {item}")
        merged = OmegaConf.merge(merged, _load_yaml(group_path))

    root_body = OmegaConf.create(OmegaConf.to_container(root, resolve=False))
    root_body.pop("defaults", None)
    merged = OmegaConf.merge(merged, root_body)
    if dot_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dot_overrides))
    OmegaConf.resolve(merged)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def config_hash(cfg: dict[str, Any]) -> str:
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True, sort_keys=True)
    return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()


def save_effective_config(path: str | Path, cfg: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(
        OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True, sort_keys=False),
        encoding="utf-8",
    )
