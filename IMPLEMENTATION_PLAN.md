# Phased Implementation Task List

This file is the active task list for the agricultural field image detection project.
We are implementing one phase at a time and pausing for inspection after each phase.

## Phase 1: Project scaffolding and environment
- [x] Create top-level project directories for source, configs, models, reports, and notebooks
- [x] Add a `pyproject.toml` dependency manifest for macOS-friendly local training
- [x] Add a minimal README with setup instructions for macOS and Apple Silicon
- [x] Add a utility module that selects `mps` when available and falls back to `cpu`
- [x] Write this plan file to disk and use it as the working checklist
- [x] Pause for inspection before moving to Phase 2

## Phase 2: Dataset validation and configuration
- [x] Add the main YAML config for dataset paths, classes, outputs, and training defaults
- [x] Add a dataset inspection script for COCO annotations
- [x] Report image counts, annotation counts, and class counts by split
- [x] Warn when schema classes have zero annotations
- [x] Validate image references and malformed boxes
- [x] Write a dataset summary artifact into `reports/`
- [x] Pause for inspection before moving to Phase 3

## Phase 3: Training baseline detector
- [x] Add `train.py`
- [x] Read config and prepare a YOLO-compatible dataset description
- [x] Auto-select device
- [ ] Train a lightweight pretrained YOLO baseline
- [x] Write outputs into a versioned run directory
- [ ] Save weights and training summary
- [x] Pause for inspection before moving to Phase 4

## Phase 4: Evaluation and error analysis
- [x] Add `evaluate.py`
- [x] Run validation/test evaluation from saved weights
- [x] Write metrics and per-class summaries into `reports/`
- [x] Save representative annotated predictions
- [x] Pause for inspection before moving to Phase 5

## Phase 5: Inference CLI for new images
- [x] Add `predict.py`
- [x] Support single-image and directory inference
- [x] Save annotated images and JSON predictions
- [x] Default outputs to `reports/predictions/...`
- [x] Verify macOS device fallback behavior
- [x] Pause for inspection before moving to Phase 6

## Phase 6: Polish and first tuning pass
- [x] Refine CLI help and config defaults
- [x] Add a simple experiment log convention
- [x] Add optional tuning controls for augmentation and thresholds
- [x] Tighten README training, evaluation, and prediction instructions
- [x] Document the future path to support `dirt`
- [x] Pause for final inspection
