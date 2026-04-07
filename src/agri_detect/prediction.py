"""Inference helpers for running detector predictions on new images."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agri_detect.config import load_config
from agri_detect.device import get_best_device
from agri_detect.experiment_log import append_experiment_log
from agri_detect.paths import PROJECT_ROOT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


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


def _resolve_device(config: dict[str, Any], requested_device: str | None) -> str:
    if requested_device:
        return requested_device

    configured_device = config.get("prediction", {}).get("device", config["training"].get("device", "auto"))
    if configured_device == "auto":
        return get_best_device()
    return str(configured_device)


def _find_latest_run_dir(runs_root: Path) -> Path:
    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        weights_dir = run_dir / "training" / "weights"
        if (weights_dir / "best.pt").exists() or (weights_dir / "last.pt").exists():
            return run_dir

    msg = f"No trained runs with weights found in {runs_root}"
    raise FileNotFoundError(msg)


def _resolve_weights_path(config: dict[str, Any], weights_path: str | Path | None) -> Path:
    if weights_path is not None:
        return Path(weights_path).resolve()

    latest_run_dir = _find_latest_run_dir(PROJECT_ROOT / config["outputs"]["runs_dir"])
    best_path = latest_run_dir / "training" / "weights" / "best.pt"
    last_path = latest_run_dir / "training" / "weights" / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path

    msg = f"No best.pt or last.pt found under {latest_run_dir / 'training' / 'weights'}"
    raise FileNotFoundError(msg)


def _collect_input_images(input_path: Path) -> list[Path]:
    resolved = input_path.resolve()
    if resolved.is_file():
        return [resolved]

    if resolved.is_dir():
        return sorted(
            [
                path
                for path in resolved.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=lambda path: path.name,
        )

    msg = f"Input path does not exist: {resolved}"
    raise FileNotFoundError(msg)


def _make_prediction_dir(config: dict[str, Any], input_path: Path) -> Path:
    predictions_root = PROJECT_ROOT / config["outputs"]["predictions_dir"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_name = input_path.stem if input_path.is_file() else input_path.name
    safe_name = input_name.replace(" ", "-").replace("/", "-")
    output_dir = predictions_root / f"{timestamp}-{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def predict_on_images(
    config_path: str | Path,
    input_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Run inference on a single image or a directory of images and save artifacts."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = "ultralytics is required to run prediction. Install project dependencies first."
        raise RuntimeError(msg) from exc

    config = load_config(config_path)
    resolved_input_path = Path(input_path).resolve()
    images = _collect_input_images(resolved_input_path)
    if not images:
        msg = f"No supported image files found in {resolved_input_path}"
        raise ValueError(msg)

    resolved_weights_path = _resolve_weights_path(config, weights_path)
    if conf == 0.25:
        conf = float(config.get("prediction", {}).get("conf", conf))
    resolved_output_dir = (
        Path(output_dir).resolve() if output_dir is not None else _make_prediction_dir(config, resolved_input_path)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(config, device)
    model = YOLO(str(resolved_weights_path))
    results = model.predict(
        source=[str(path) for path in images],
        device=resolved_device,
        conf=conf,
        verbose=False,
        save=False,
    )

    image_summaries: list[dict[str, Any]] = []
    for image_path, result in zip(images, results, strict=False):
        annotated_path = resolved_output_dir / image_path.name
        result.save(filename=str(annotated_path))
        image_summary = {
            "source_image": str(image_path),
            "annotated_image": str(annotated_path),
            "detections": _json_ready(result.summary(normalize=True)),
        }
        prediction_json_path = resolved_output_dir / f"{image_path.stem}.json"
        _write_json(prediction_json_path, image_summary)
        image_summary["prediction_json"] = str(prediction_json_path)
        image_summaries.append(image_summary)

    prediction_summary = {
        "project_name": config["project"]["name"],
        "config_path": str(config["_config_path"]),
        "input_path": str(resolved_input_path),
        "image_count": len(image_summaries),
        "weights_path": str(resolved_weights_path),
        "device": resolved_device,
        "confidence_threshold": conf,
        "output_dir": str(resolved_output_dir),
        "images": image_summaries,
    }
    _write_json(resolved_output_dir / "predictions_summary.json", prediction_summary)
    append_experiment_log(config["outputs"]["experiment_log"], "predict", prediction_summary)
    return prediction_summary
