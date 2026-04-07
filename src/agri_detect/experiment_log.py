"""Simple experiment logging helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agri_detect.paths import PROJECT_ROOT


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def append_experiment_log(log_path: str | Path, event_type: str, payload: dict[str, Any]) -> Path:
    """Append a single JSONL experiment event to the configured log path."""
    resolved_log_path = Path(log_path)
    if not resolved_log_path.is_absolute():
        resolved_log_path = PROJECT_ROOT / resolved_log_path

    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "payload": _json_ready(payload),
    }
    with resolved_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return resolved_log_path
