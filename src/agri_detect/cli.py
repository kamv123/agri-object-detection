"""Small helpers for future CLI entrypoints."""

from __future__ import annotations

import argparse

from agri_detect.device import describe_device


def build_base_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with shared project messaging."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.epilog = describe_device()
    return parser
