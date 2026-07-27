from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("output_dir", "outputs"))


def ensure_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = output_dir(cfg)
    dirs = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "matrices": root / "matrices",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")
