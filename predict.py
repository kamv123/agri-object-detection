"""CLI entrypoint for running detector inference on new images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agri_detect.cli import build_base_parser
from agri_detect.prediction import predict_on_images


def main() -> int:
    parser = build_base_parser("Run YOLO inference on one image or a directory of images.")
    parser.add_argument("input", help="Path to a single image or a directory of images.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the main project config file.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit weights path. Defaults to the latest trained run's best weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to reports/predictions/<timestamp>-<name>.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit inference device, such as cpu or mps.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    args = parser.parse_args()

    summary = predict_on_images(
        args.config,
        args.input,
        weights_path=args.weights,
        output_dir=args.output_dir,
        device=args.device,
        conf=args.conf,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
