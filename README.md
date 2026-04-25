# Agricultural Field Image Detection

Local Python project for training and running an object detector on agricultural field images.
Version 1 targets `weed` and `wheat` detection from the COCO-format dataset already present in `Data/`.

## Project status

Phases 1 through 5 are implemented: dataset inspection, training prep, evaluation, and prediction CLIs are all available.
The next phase is focused on polish, logging, and tuning defaults.

## macOS setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade packaging tools:

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

3. Install dependencies:

```bash
pip install --no-build-isolation -e .
```

## Apple Silicon notes

- PyTorch can use Metal Performance Shaders through the `mps` device on supported Macs.
- The project will automatically use `mps` when available and fall back to `cpu` otherwise.
- Training speed on macOS is usually best with smaller baseline models first.
- Evaluation defaults to `cpu` in config because validation on `mps` proved less stable on this machine.

## CLI workflow

Inspect the dataset and write a summary:

```bash
python inspect_dataset.py
```

Prepare a versioned YOLO dataset bundle without starting training:

```bash
python train.py --prepare-only
```

Start baseline training:

```bash
python train.py
```

Evaluate saved weights with validation and test reports:

```bash
python evaluate.py --weights yolo11n.pt
```

Run prediction on one image:

```bash
python predict.py path/to/image.jpg --weights yolo11n.pt
```

Run prediction on a directory:

```bash
python predict.py path/to/images/ --weights yolo11n.pt
```

## Outputs and logging

- Dataset inspection writes `reports/dataset_summary.json`
- Training writes versioned run folders under `reports/runs/`
- Evaluation writes metrics and annotated samples under each run's `evaluation/` directory
- Prediction writes timestamped output folders under `reports/predictions/`
- Training, evaluation, and prediction append JSONL events to `reports/experiment_log.jsonl`

## Tuning defaults

The main config in `configs/project.yaml` now includes:

- training augmentation defaults
- evaluation device and sample-count defaults
- prediction confidence and device defaults

These can be changed in config before running the scripts, or overridden via CLI flags where available.

## Future `dirt` support

The current active class set is `weed` and `wheat`.
The future path for `dirt` is documented in `configs/project.yaml`:

- add `dirt` to `schema_classes` once labeled examples exist
- decide whether `dirt` is a first-class object label or a hard-negative/background concept
- rerun inspection, training, evaluation, and prediction after enabling it

## Current layout

- `Data/`: source dataset
- `configs/`: dataset and training configuration
- `models/`: saved weights and exported artifacts
- `notebooks/`: optional exploration notebooks
- `reports/`: dataset summaries, metrics, and predictions
- `src/`: Python package and CLI scripts

## Planned scripts

- `train.py`: train a baseline detector
- `evaluate.py`: evaluate saved weights
- `predict.py`: run inference on new images
