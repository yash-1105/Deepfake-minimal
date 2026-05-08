<div align="center">

# 🕵️ Deepfake-Minimal

**A lightweight, pose-based deepfake detection pipeline using MediaPipe + PyTorch**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897B?style=flat-square)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**Deepfake-Minimal** is a minimal yet complete deepfake detection system that works by analyzing **body pose motion** rather than pixel-level facial features. The core idea: deepfakes introduce subtle unnatural artifacts in full-body pose dynamics that an MLP or LSTM classifier can learn to detect.

### How It Works

```
Video Input
    │
    ▼
┌─────────────────────┐
│  Pose Extraction     │  ← MediaPipe detects 33 body landmarks per frame
│  (extract_pose.py)   │
└─────────┬───────────┘
          │  (T × 33 × 3) pose sequence
          ▼
┌─────────────────────┐
│  Feature Extraction  │  ← Joint angles, velocities, accelerations
│ (extract_features.py)│
└─────────┬───────────┘
          │  (T × F) feature matrix
          ▼
┌─────────────────────┐
│  MLP / LSTM Model    │  ← Trained on real vs. synthetic-fake pose data
│   (train_model.py)   │
└─────────┬───────────┘
          │
          ▼
    REAL / FAKE + Confidence Score
```

---

## 🗂️ Project Structure

```
Deepfake-minimal/
│
├── src/
│   ├── extract_pose.py       # Extract MediaPipe pose landmarks from frame directories
│   ├── extract_features.py   # Compute motion features from pose sequences
│   ├── synth_dataset.py      # Generate synthetic fake poses for training
│   ├── train_model.py        # Train MLP or LSTM classifier
│   ├── evaluate_fixed.py     # Evaluate model accuracy & AUC on a feature dataset
│   └── run_inference.py      # Run inference on a video file
│
├── models/                   # Saved model weights (not tracked in git)
├── demos/                    # Demo videos (not tracked in git)
├── results/                  # Output/annotated videos (not tracked in git)
│
├── requirements.txt          # Python dependencies
├── INSTALL_SETUP_WINDOWS_MAC.md
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yash-1105/Deepfake-minimal.git
cd Deepfake-minimal
```

### 2. Create a Virtual Environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch install may vary by platform. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the right command if the above doesn't work.

---

## 🚀 Usage

### Step 1 — Extract Poses from Frame Directories

Extract MediaPipe body pose landmarks from a folder of image frames.

```bash
python src/extract_pose.py \
  --input  /path/to/frames_root \
  --output models/features_run \
  --max-frames 120 \
  --stride 1
```

| Argument | Description |
|---|---|
| `--input` | Root folder where each subfolder is one video's frames (`.jpg`/`.png`) |
| `--output` | Where to save `.npz` pose files |
| `--max-frames` | Maximum frames to process per video (default: `120`) |
| `--stride` | Frame step (default: `1`, use `2` to skip every other frame) |

---

### Step 2 — Generate Synthetic Fake Poses

Create fake training samples by perturbing real pose sequences.

```bash
python src/synth_dataset.py \
  --pose-folder models/features_run \
  --out         models/features_run/fake \
  --num-samples 500
```

| Argument | Description |
|---|---|
| `--pose-folder` | Folder containing real `.npz` pose files |
| `--out` | Output folder for synthetic fake `.npz` files |
| `--num-samples` | Number of fake samples to generate (default: `20`) |

---

### Step 3 — Extract Motion Features

Compute joint-angle and velocity-based features from the raw pose sequences.

```bash
python src/extract_features.py \
  --pose-dir models/features_run \
  --out-dir  models/features_extracted
```

---

### Step 4 — Train the Model

Train an MLP (fast) or LSTM (sequence-aware) classifier.

```bash
# Train MLP (recommended for quick experiments)
python src/train_model.py \
  --features models/features_extracted \
  --out       models/model.pt \
  --model-type mlp \
  --epochs 50

# Train LSTM (better for temporal patterns)
python src/train_model.py \
  --features models/features_extracted \
  --out       models/model_lstm.pt \
  --model-type lstm \
  --epochs 50
```

| Argument | Description |
|---|---|
| `--features` | Path to extracted features folder |
| `--out` | Path to save the trained model weights |
| `--model-type` | `mlp` or `lstm` (default: `mlp`) |
| `--epochs` | Number of training epochs (default: `50`) |

---

### Step 5 — Evaluate the Model

Compute accuracy and AUC on a feature dataset.

```bash
python src/evaluate_fixed.py \
  --model      models/model.pt \
  --features   models/features_extracted \
  --model-type mlp
```

**Example output:**
```
Samples: 650  FeatureDim: 99
Accuracy: 0.8923
AUC: 0.9341
```

---

### Step 6 — Run Inference on a Video

Detect whether a video is real or deepfake.

```bash
# Basic inference
python src/run_inference.py \
  --input      demos/your_video.mp4 \
  --model      models/model.pt \
  --model-type mlp

# With annotated output video
python src/run_inference.py \
  --input         demos/your_video.mp4 \
  --model         models/model.pt \
  --model-type    mlp \
  --out           results/annotated.mp4 \
  --fake-threshold 0.35

# Interactive file picker (no --input needed)
python src/run_inference.py \
  --model      models/model.pt \
  --model-type mlp \
  --file-dialog
```

| Argument | Description |
|---|---|
| `--input` | Path to input video file |
| `--model` | Path to trained model `.pt` file |
| `--model-type` | `mlp` or `lstm` (must match training) |
| `--out` | *(Optional)* Path to save annotated output video |
| `--max-frames` | Max frames to process (default: `120`) |
| `--stride` | Frame sampling stride (default: `1`) |
| `--fake-threshold` | Probability threshold to classify as FAKE (default: `0.35`) |
| `--file-dialog` | Open OS file picker to choose video interactively |
| `--device` | `cpu`, `cuda`, or `auto` (default: `auto`) |

**Example output:**
```
Prediction : FAKE
Score      : REAL=0.2341  FAKE=0.7659
Confidence : 0.7659
Inference time: 1.43 sec
```

---

## 🔧 Tips & Tricks

- **Lower `--fake-threshold`** (e.g., `0.25`) → more sensitive to fakes, more false positives  
- **Higher `--fake-threshold`** (e.g., `0.50`) → only flag confident fakes, fewer false positives  
- Use `--stride 2` on long videos to speed up processing  
- LSTM generally gives better results on high-motion videos; MLP is faster  

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.9.1 | Neural network training & inference |
| `mediapipe` | 0.10.21 | Body pose landmark detection |
| `opencv-python` | 4.11.x | Video I/O and frame processing |
| `numpy` | 1.26.x | Numerical computation |
| `scikit-learn` | 1.7.x | Evaluation metrics (accuracy, AUC) |
| `tqdm` | 4.67.x | Progress bars |
| `matplotlib` | 3.10.x | Plotting and visualization |

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ for deepfake research
</div>
