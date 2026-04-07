"""Evaluation helpers for trained YOLO detection runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agri_detect.config import load_config
from agri_detect.device import get_best_device
from agri_detect.experiment_log import append_experiment_log
from agri_detect.paths import PROJECT_ROOT


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
        except (ValueError, TypeError):
            return str(value)
    return value


def _resolve_device(config: dict[str, Any], requested_device: str | None) -> str:
    if requested_device:
        return requested_device

    configured_eval_device = config.get("evaluation", {}).get("device")
    if configured_eval_device:
        return str(configured_eval_device)

    configured_device = config["training"].get("device", "auto")
    if configured_device != "auto":
        return str(configured_device)

    best_device = get_best_device()
    # Validation is smaller than training, so prefer CPU stability by default on macOS MPS.
    if best_device == "mps":
        return "cpu"
    return best_device


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


def _resolve_run_dir(run_dir: str | Path | None, weights_path: str | Path | None) -> Path | None:
    if run_dir is not None:
        return Path(run_dir).resolve()

    if weights_path is not None:
        weights = Path(weights_path).resolve()
        if weights.parent.name == "weights" and weights.parent.parent.name == "training":
            return weights.parent.parent.parent

    return None


def _resolve_weights_path(run_dir: Path | None, weights_path: str | Path | None) -> Path:
    if weights_path is not None:
        return Path(weights_path).resolve()

    if run_dir is None:
        msg = "A run directory or weights path is required for evaluation."
        raise ValueError(msg)

    best_path = run_dir / "training" / "weights" / "best.pt"
    last_path = run_dir / "training" / "weights" / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path

    msg = f"No best.pt or last.pt found under {run_dir / 'training' / 'weights'}"
    raise FileNotFoundError(msg)


def _resolve_dataset_yaml(run_dir: Path | None, dataset_yaml: str | Path | None) -> Path:
    if dataset_yaml is not None:
        return Path(dataset_yaml).resolve()

    if run_dir is None:
        msg = "A run directory or dataset YAML path is required for evaluation."
        raise ValueError(msg)

    resolved = run_dir / "prepared_dataset" / "dataset.yaml"
    if not resolved.exists():
        msg = f"Prepared dataset description not found at {resolved}"
        raise FileNotFoundError(msg)
    return resolved


def _collect_sample_images(dataset_yaml_path: Path, split_name: str, sample_count: int) -> list[Path]:
    images_dir = dataset_yaml_path.parent / "images" / split_name
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )[:sample_count]


def _run_split_metrics(
    model: Any,
    *,
    dataset_yaml_path: Path,
    split_name: str,
    device: str,
    evaluation_dir: Path,
) -> dict[str, Any]:
    save_dir = evaluation_dir / f"{split_name}_metrics"
    metrics = model.val(
        data=str(dataset_yaml_path),
        split=split_name,
        device=device,
        project=str(evaluation_dir),
        name=f"{split_name}_metrics",
        exist_ok=True,
        plots=True,
    )

    split_summary = {
        "split": split_name,
        "save_dir": str(save_dir),
        "results": _json_ready(getattr(metrics, "results_dict", {})),
        "per_class": _json_ready(metrics.summary()),
    }
    _write_json(evaluation_dir / f"{split_name}_metrics.json", split_summary)
    return split_summary


def _save_prediction_samples(
    model: Any,
    *,
    dataset_yaml_path: Path,
    split_name: str,
    device: str,
    evaluation_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    sample_images = _collect_sample_images(dataset_yaml_path, split_name, sample_count)
    predictions_dir = evaluation_dir / "predictions" / split_name
    predictions_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=[str(path) for path in sample_images],
        device=device,
        verbose=False,
        save=False,
    )

    samples: list[dict[str, Any]] = []
    for source_path, result in zip(sample_images, results, strict=False):
        output_path = predictions_dir / source_path.name
        result.save(filename=str(output_path))
        samples.append(
            {
                "source_image": str(source_path),
                "annotated_prediction": str(output_path),
                "detections": _json_ready(result.summary(normalize=True)),
            }
        )

    prediction_summary = {
        "split": split_name,
        "sample_count": len(samples),
        "predictions_dir": str(predictions_dir),
        "samples": samples,
    }
    _write_json(evaluation_dir / f"{split_name}_predictions.json", prediction_summary)
    return prediction_summary


def evaluate_trained_detector(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    dataset_yaml: str | Path | None = None,
    sample_count: int = 5,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate trained weights on validation and test splits and save artifacts."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = "ultralytics is required to evaluate. Install project dependencies first."
        raise RuntimeError(msg) from exc

    config = load_config(config_path)
    resolved_run_dir = _resolve_run_dir(run_dir, weights_path)
    if resolved_run_dir is None:
        resolved_run_dir = _find_latest_run_dir(PROJECT_ROOT / config["outputs"]["runs_dir"])

    resolved_weights_path = _resolve_weights_path(resolved_run_dir, weights_path)
    resolved_dataset_yaml = _resolve_dataset_yaml(resolved_run_dir, dataset_yaml)
    if sample_count == 5:
        sample_count = int(config.get("evaluation", {}).get("sample_count", sample_count))
    device = _resolve_device(config, device)

    evaluation_dir = resolved_run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(resolved_weights_path))
    split_summaries: dict[str, Any] = {}

    for split_name in ("valid", "test"):
        yolo_split = "val" if split_name == "valid" else split_name
        metrics_summary = _run_split_metrics(
            model,
            dataset_yaml_path=resolved_dataset_yaml,
            split_name=yolo_split,
            device=device,
            evaluation_dir=evaluation_dir,
        )
        predictions_summary = _save_prediction_samples(
            model,
            dataset_yaml_path=resolved_dataset_yaml,
            split_name=yolo_split,
            device=device,
            evaluation_dir=evaluation_dir,
            sample_count=sample_count,
        )
        split_summaries[split_name] = {
            "metrics": metrics_summary,
            "predictions": predictions_summary,
        }

    evaluation_summary = {
        "project_name": config["project"]["name"],
        "config_path": str(config["_config_path"]),
        "run_dir": str(resolved_run_dir),
        "weights_path": str(resolved_weights_path),
        "dataset_yaml": str(resolved_dataset_yaml),
        "device": device,
        "evaluation_dir": str(evaluation_dir),
        "splits": split_summaries,
    }
    _write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)
    append_experiment_log(config["outputs"]["experiment_log"], "evaluate", evaluation_summary)
    return evaluation_summary
