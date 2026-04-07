"""CLI entrypoint for evaluating trained detector weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agri_detect.cli import build_base_parser
from agri_detect.evaluation import evaluate_trained_detector


def main() -> int:
    parser = build_base_parser("Evaluate saved YOLO weights and export metrics plus samples.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the main project config file.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory containing prepared data and training outputs.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit weights path. Defaults to the latest run's best weights.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Optional explicit dataset YAML path if it is not under the run directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="How many representative images to annotate per split.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit evaluation device, such as cpu or mps.",
    )
    args = parser.parse_args()

    summary = evaluate_trained_detector(
        args.config,
        run_dir=args.run_dir,
        weights_path=args.weights,
        dataset_yaml=args.dataset_yaml,
        sample_count=args.sample_count,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
