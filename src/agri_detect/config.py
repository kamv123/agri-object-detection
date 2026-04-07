"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agri_detect.paths import PROJECT_ROOT


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and return normalized project paths."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        msg = f"Expected config mapping in {path}"
        raise ValueError(msg)

    config["_config_path"] = path
    config["_project_root"] = PROJECT_ROOT
    return config


def resolve_from_project(*parts: str | Path) -> Path:
    """Resolve a path from the project root."""
    return PROJECT_ROOT.joinpath(*parts)
