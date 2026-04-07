"""Dataset inspection utilities for COCO-formatted object detection data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agri_detect.config import load_config
from agri_detect.paths import PROJECT_ROOT


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    return x, y, width, height


def inspect_coco_dataset(config_path: str | Path) -> dict[str, Any]:
    """Inspect a COCO dataset and return counts, warnings, and validation issues."""
    config = load_config(config_path)
    dataset_cfg = config["dataset"]

    dataset_root = PROJECT_ROOT / dataset_cfg["root"]
    image_dirs = dataset_cfg["image_dirs"]
    annotation_paths = {
        "train": dataset_root / dataset_cfg["train_annotations"],
        "valid": dataset_root / dataset_cfg["valid_annotations"],
        "test": dataset_root / dataset_cfg["test_annotations"],
    }

    summary: dict[str, Any] = {
        "project_name": config["project"]["name"],
        "dataset_root": str(dataset_root),
        "active_classes": dataset_cfg["active_classes"],
        "excluded_classes": dataset_cfg["excluded_classes"],
        "splits": {},
        "overall": {
            "images": 0,
            "annotations": 0,
            "class_counts": {},
        },
        "warnings": [],
    }

    overall_counts: Counter[str] = Counter()
    schema_classes = list(dataset_cfg["schema_classes"])

    for split_name, annotation_path in annotation_paths.items():
        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = coco.get("categories", [])
        category_map = {item["id"]: item["name"] for item in categories}
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        image_lookup = {image["id"]: image for image in images}

        split_counts: Counter[str] = Counter()
        missing_images: list[str] = []
        malformed_annotations: list[dict[str, Any]] = []

        for image in images:
            image_path = dataset_root / image_dirs[split_name] / image["file_name"]
            if not image_path.exists():
                missing_images.append(image["file_name"])

        for annotation in annotations:
            category_name = category_map.get(annotation.get("category_id"), "unknown")
            split_counts[category_name] += 1
            bbox = _normalize_bbox(annotation.get("bbox"))

            if bbox is None:
                malformed_annotations.append(
                    {
                        "annotation_id": annotation.get("id"),
                        "reason": "bbox_missing_or_invalid",
                    }
                )
                continue

            x, y, width, height = bbox
            image = image_lookup.get(annotation.get("image_id"))
            if image is None:
                malformed_annotations.append(
                    {
                        "annotation_id": annotation.get("id"),
                        "reason": "image_reference_missing",
                    }
                )
                continue

            image_width = image.get("width")
            image_height = image.get("height")

            if width <= 0 or height <= 0:
                malformed_annotations.append(
                    {
                        "annotation_id": annotation.get("id"),
                        "reason": "non_positive_box_size",
                    }
                )
                continue

            if (
                image_width is not None
                and image_height is not None
                and (x < 0 or y < 0 or x + width > image_width or y + height > image_height)
            ):
                malformed_annotations.append(
                    {
                        "annotation_id": annotation.get("id"),
                        "reason": "box_outside_image_bounds",
                    }
                )

        summary["splits"][split_name] = {
            "images": len(images),
            "annotations": len(annotations),
            "class_counts": dict(sorted(split_counts.items())),
            "missing_images": missing_images,
            "malformed_annotation_count": len(malformed_annotations),
            "malformed_annotations_preview": malformed_annotations[:10],
        }

        overall_counts.update(split_counts)
        summary["overall"]["images"] += len(images)
        summary["overall"]["annotations"] += len(annotations)

    zero_annotation_classes = [
        class_name for class_name in schema_classes if overall_counts.get(class_name, 0) == 0
    ]

    if zero_annotation_classes:
        summary["warnings"].append(
            {
                "type": "zero_annotation_classes",
                "classes": zero_annotation_classes,
                "message": "Some schema classes are present in annotations metadata but unused.",
            }
        )

    if dataset_cfg["excluded_classes"]:
        summary["warnings"].append(
            {
                "type": "excluded_classes",
                "classes": dataset_cfg["excluded_classes"],
                "message": "Excluded classes are intentionally omitted from the v1 active class set.",
            }
        )

    summary["overall"]["class_counts"] = dict(sorted(overall_counts.items()))
    return summary


def write_dataset_summary(config_path: str | Path) -> Path:
    """Write the dataset inspection summary to the configured report path."""
    config = load_config(config_path)
    summary = inspect_coco_dataset(config_path)
    output_path = PROJECT_ROOT / config["outputs"]["dataset_summary"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path
