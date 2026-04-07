"""Training helpers for preparing and running a baseline YOLO detector."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agri_detect.config import load_config
from agri_detect.device import get_best_device
from agri_detect.experiment_log import append_experiment_log
from agri_detect.paths import PROJECT_ROOT


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    return x, y, width, height


def _select_training_device(training_cfg: dict[str, Any]) -> str:
    requested_device = training_cfg.get("device", "auto")
    if requested_device == "auto":
        return get_best_device()
    return str(requested_device)


def _make_run_dir(config: dict[str, Any]) -> Path:
    runs_root = PROJECT_ROOT / config["outputs"]["runs_dir"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = str(config["training"]["model"]).replace("/", "-").replace(".", "-")
    run_dir = runs_root / f"{timestamp}-{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _link_image(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    destination_path.symlink_to(source_path.resolve())


def _prepare_yolo_dataset(config: dict[str, Any], run_dir: Path) -> tuple[Path, dict[str, Any]]:
    dataset_cfg = config["dataset"]
    dataset_root = PROJECT_ROOT / dataset_cfg["root"]
    prepared_root = run_dir / "prepared_dataset"
    images_root = prepared_root / "images"
    labels_root = prepared_root / "labels"

    active_classes = list(dataset_cfg["active_classes"])
    active_class_lookup = {class_name: index for index, class_name in enumerate(active_classes)}
    split_key_map = {"train": "train_annotations", "valid": "valid_annotations", "test": "test_annotations"}

    prep_summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "active_classes": active_classes,
        "splits": {},
    }

    for split_name, annotation_key in split_key_map.items():
        annotation_path = dataset_root / dataset_cfg[annotation_key]
        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])
        category_map = {item["id"]: item["name"] for item in categories}
        image_lookup = {image["id"]: image for image in images}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for annotation in annotations:
            annotations_by_image[annotation.get("image_id")].append(annotation)

        split_image_dir = images_root / split_name
        split_label_dir = labels_root / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        split_summary = {
            "images": len(images),
            "annotations_total": len(annotations),
            "annotations_written": 0,
            "annotations_skipped": 0,
            "images_linked": 0,
        }

        for image in images:
            source_image_path = dataset_root / dataset_cfg["image_dirs"][split_name] / image["file_name"]
            linked_image_path = split_image_dir / image["file_name"]
            _link_image(source_image_path, linked_image_path)
            split_summary["images_linked"] += 1

            label_lines: list[str] = []
            image_width = image.get("width")
            image_height = image.get("height")

            for annotation in annotations_by_image.get(image["id"], []):
                class_name = category_map.get(annotation.get("category_id"))
                if class_name not in active_class_lookup:
                    split_summary["annotations_skipped"] += 1
                    continue

                bbox = _normalize_bbox(annotation.get("bbox"))
                if bbox is None or not image_width or not image_height:
                    split_summary["annotations_skipped"] += 1
                    continue

                x, y, width, height = bbox
                if width <= 0 or height <= 0:
                    split_summary["annotations_skipped"] += 1
                    continue

                x_center = (x + (width / 2.0)) / float(image_width)
                y_center = (y + (height / 2.0)) / float(image_height)
                normalized_width = width / float(image_width)
                normalized_height = height / float(image_height)

                if not all(
                    0.0 <= value <= 1.0
                    for value in (x_center, y_center, normalized_width, normalized_height)
                ):
                    split_summary["annotations_skipped"] += 1
                    continue

                class_index = active_class_lookup[class_name]
                label_lines.append(
                    f"{class_index} {x_center:.6f} {y_center:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )
                split_summary["annotations_written"] += 1

            label_path = split_label_dir / f"{Path(image['file_name']).stem}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

        prep_summary["splits"][split_name] = split_summary

    dataset_yaml_path = prepared_root / "dataset.yaml"
    dataset_yaml = {
        "path": str(prepared_root),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {index: class_name for index, class_name in enumerate(active_classes)},
        "nc": len(active_classes),
    }
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    return dataset_yaml_path, prep_summary


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def prepare_training_run(config_path: str | Path) -> dict[str, Any]:
    """Create a versioned run directory and YOLO-ready dataset bundle."""
    config = load_config(config_path)
    run_dir = _make_run_dir(config)
    dataset_yaml_path, prep_summary = _prepare_yolo_dataset(config, run_dir)
    device = _select_training_device(config["training"])

    preparation_summary = {
        "project_name": config["project"]["name"],
        "config_path": str(config["_config_path"]),
        "run_dir": str(run_dir),
        "dataset_yaml": str(dataset_yaml_path),
        "device": device,
        "training": dict(config["training"]),
        "dataset_preparation": prep_summary,
    }
    _write_json(run_dir / "dataset_prep_summary.json", preparation_summary)
    return preparation_summary


def train_baseline_detector(
    config_path: str | Path,
    *,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Prepare data, optionally train a YOLO detector, and write a run summary."""
    preparation_summary = prepare_training_run(config_path)
    run_dir = Path(preparation_summary["run_dir"])

    if prepare_only:
        preparation_summary["status"] = "prepared"
        _write_json(run_dir / "training_summary.json", preparation_summary)
        return preparation_summary

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = "ultralytics is required to train. Install project dependencies first."
        raise RuntimeError(msg) from exc

    training_cfg = preparation_summary["training"]
    model = YOLO(training_cfg["model"])
    training_output_dir = run_dir / "training"

    train_result = model.train(
        data=preparation_summary["dataset_yaml"],
        imgsz=training_cfg["image_size"],
        batch=training_cfg["batch_size"],
        epochs=training_cfg["epochs"],
        patience=training_cfg["patience"],
        workers=training_cfg["workers"],
        pretrained=training_cfg["pretrained"],
        device=preparation_summary["device"],
        project=str(run_dir),
        name="training",
        exist_ok=True,
        **training_cfg.get("augment", {}),
    )

    results_dict = getattr(train_result, "results_dict", None)
    training_summary = {
        **preparation_summary,
        "status": "trained",
        "training_output_dir": str(training_output_dir),
        "best_weights": str(training_output_dir / "weights" / "best.pt"),
        "last_weights": str(training_output_dir / "weights" / "last.pt"),
        "results": results_dict if isinstance(results_dict, dict) else {},
    }
    _write_json(run_dir / "training_summary.json", training_summary)
    append_experiment_log(config["outputs"]["experiment_log"], "train", training_summary)
    return training_summary
