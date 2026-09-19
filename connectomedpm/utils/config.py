"""Config loading with dotted overrides.

Usage:
    cfg = load_config("configs/experiment/h1_pilot.yaml", overrides=["router.seed=7"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from .io import read_yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like path=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return out


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    cfg = read_yaml(path) or {}
    if "extends" in cfg:
        parent_path = (Path(path).parent / cfg["extends"]).resolve()
        parent = load_config(parent_path)
        cfg = deep_merge(parent, {k: v for k, v in cfg.items() if k != "extends"})
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("_config_path", str(path))
    return cfg


def resolve_path(cfg: dict, value: str) -> Path:
    """Resolve a possibly relative path against the repo root (config parent's parent)."""
    p = Path(value)
    if p.is_absolute():
        return p
    root = Path(cfg.get("repo_root", "."))
    return (root / p).resolve()
