#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/config.yaml")
CONFIG_DIR = Path("configs")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid yaml object: {path}")
    return data


def default_group(config: dict[str, Any], group_name: str) -> str:
    for item in config.get("defaults", []):
        if isinstance(item, dict) and group_name in item:
            value = item[group_name]
            if not isinstance(value, str):
                raise SystemExit(f"invalid defaults entry for {group_name}: {value!r}")
            return value
    raise SystemExit(f"missing defaults entry: {group_name}")


def enabled_providers(teacher: dict[str, Any]) -> list[dict[str, Any]]:
    providers = teacher.get("providers", [])
    if not isinstance(providers, list):
        raise SystemExit("teacher.providers must be a list")
    enabled = []
    for provider in providers:
        if not isinstance(provider, dict):
            raise SystemExit(f"invalid teacher provider: {provider!r}")
        if float(provider.get("weight", 0.0) or 0.0) > 0:
            enabled.append(provider)
    return enabled


def main() -> None:
    root_config = load_yaml(CONFIG_PATH)
    teacher_cfg = load_yaml(CONFIG_DIR / "teacher.yaml")["teacher"]

    providers = enabled_providers(teacher_cfg)
    if not providers:
        print("none enabled")
        return
    for provider in providers:
        print(
            f"{provider['name']} model={provider['model']} "
            f"weight={provider['weight']} api_key_env={provider['api_key_env']}"
        )


if __name__ == "__main__":
    main()
