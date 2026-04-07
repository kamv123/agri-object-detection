"""CLI entrypoint for inspecting the COCO dataset used by this project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agri_detect.cli import build_base_parser
from agri_detect.dataset_inspector import inspect_coco_dataset, write_dataset_summary


def main() -> int:
    parser = build_base_parser("Inspect the local COCO dataset and write a summary report.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the main project config file.",
    )
    args = parser.parse_args()

    summary = inspect_coco_dataset(args.config)
    output_path = write_dataset_summary(args.config)

    console_summary = {
        "dataset_root": summary["dataset_root"],
        "active_classes": summary["active_classes"],
        "overall": summary["overall"],
        "warnings": summary["warnings"],
        "report_path": str(output_path),
    }
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
