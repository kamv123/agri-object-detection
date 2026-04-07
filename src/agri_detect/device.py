"""Device selection helpers for local training and inference."""

from __future__ import annotations

import platform


def get_best_device() -> str:
    """Return the preferred torch device for the current machine."""
    try:
        import torch
    except ImportError:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if (
        platform.system() == "Darwin"
        and mps_backend is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def describe_device() -> str:
    """Return a short human-readable description of the selected device."""
    device = get_best_device()
    if device == "mps":
        return "Using Apple Metal Performance Shaders (mps)."
    return "Using cpu."
