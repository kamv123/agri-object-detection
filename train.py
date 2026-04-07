"""CLI entrypoint for preparing and training the baseline detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agri_detect.cli import build_base_parser
from agri_detect.training import train_baseline_detector


def main() -> int:
    parser = build_base_parser("Prepare a YOLO dataset bundle and train the baseline detector.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the main project config file.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the run directory and YOLO dataset bundle without starting training.",
    )
    args = parser.parse_args()

    summary = train_baseline_detector(args.config, prepare_only=args.prepare_only)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
