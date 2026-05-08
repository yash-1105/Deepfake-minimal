# Deepfake-Minimal Project Handoff Guide

This document explains what this project does, how to run it, and how to use it during a demo/showcase.

## 1) Project Summary

This project predicts whether an input video/image is:
- `REAL`
- `FAKE`

Main entry script:
- `src/run_inference.py`

Current default model:
- `models/model.pt` (MLP mode)

## 2) Current Project Layout (Important)

Trimmed project (intentionally cleaned):
- `src/` -> source scripts
- `models/model.pt` -> main trained model
- `models/features_run/` -> feature dataset used for training/evaluation
- `demos/` -> sample demo media
- `results/` -> generated output videos
- `requirements.txt` -> dependencies

Note:
- Raw dataset/tooling folders were removed to keep the project lightweight for sharing and showcasing.
- You can still run inference, evaluate, and train from existing `models/features_run`.

## 3) Standard Demo Flow (Recommended)

From project root (venv active):

```bash
python src/run_inference.py --file-dialog --model "models/model.pt" --model-type mlp --out "results/showcase_demo.mp4"
```

What happens:
- File picker opens
- User selects a video/image
- Script predicts real/fake
- Annotated output video is saved in `results/showcase_demo.mp4`

## 4) Other Useful Run Modes

### A) Run with a direct file path

```bash
python src/run_inference.py --input "/absolute/path/to/video.mp4" --model "models/model.pt" --model-type mlp --out "results/custom_output.mp4"
```

### B) Run using `demos/` terminal selector

```bash
python src/run_inference.py --model "models/model.pt" --model-type mlp --out "results/demo_output.mp4"
```

At prompt:
- Press `Enter` for first demo file
- Or type index (`0`, `1`, etc.)
- Or paste full file path

## 5) How to Add New Videos

Two options:

### Option 1 (Easiest): Use file dialog

- Keep video anywhere on laptop.
- Run the `--file-dialog` command and select it.

### Option 2: Put videos into `demos/`

- Copy `.mp4/.avi/.mov/.mkv` (or `.jpg/.png`) into `demos/`
- Run selector mode (no `--input`)
- Choose by index

## 6) Where Results Go

Output videos are written to the path passed in `--out`.

Examples:
- `results/showcase_demo.mp4`
- `results/custom_output.mp4`

## 7) Interpreting Output

Terminal output includes:
- `Prediction: REAL` or `Prediction: FAKE`
- `Score REAL=... FAKE=...`
- `Confidence: ...`
- `Inference time: ... sec`

Output media also has overlay text with class + confidence.

## 8) Evaluation Command

Evaluate the saved model on current feature set:

```bash
python src/evaluate_fixed.py --model "models/model.pt" --features "models/features_run" --model-type mlp
```

## 9) Training Command (Using Existing Features)

Train a new model from existing features:

```bash
python src/train_model.py --features "models/features_run" --out "models/model_new.pt" --epochs 12 --batch 32 --model mlp
```

Then run inference with the new model:

```bash
python src/run_inference.py --file-dialog --model "models/model_new.pt" --model-type mlp --out "results/new_model_demo.mp4"
```

## 10) Demo-Day Checklist

- Activate venv
- Confirm imports:

```bash
python -c "import numpy, cv2, mediapipe, torch; print('OK')"
```

- Keep 1-2 short sample videos ready
- Run `--file-dialog` command
- Show both terminal output and generated video in `results/`

## 11) Known Constraints

- This is a lightweight handoff build, not full research pipeline.
- Raw collection/preprocessing tool folders were removed.
- Training currently depends on existing prepared features in `models/features_run`.

## 12) If Something Breaks Quickly

Use the setup document:
- `INSTALL_SETUP_WINDOWS_MAC.md`

Most common recovery:
- Recreate venv
- Reinstall from `requirements.txt`
- Apply OpenCV/NumPy fix from setup doc if needed
