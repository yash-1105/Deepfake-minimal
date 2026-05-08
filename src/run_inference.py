#!/usr/bin/env python3
"""
Run inference on a video or image using MLP or LSTM model.
If --input is omitted, script lets you pick from demos/ interactively.

Usage Examples:
# Interactive mode
python src/run_inference.py --model models/model_surreal_lstm.pt --model-type lstm --out results/demo.mp4

# Direct file
python src/run_inference.py --input path/to/video.mp4 --model models/model_surreal_lstm.pt --model-type lstm --out results/demo.mp4
"""

import argparse
import os
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe required. Install with: pip install mediapipe")

mp_pose = mp.solutions.pose


# ----------------------------------------------------------
# Per-frame feature extraction
# ----------------------------------------------------------
def compute_frame_features(pose_seq):
    T = pose_seq.shape[0]
    pos = pose_seq.reshape(T, -1)

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    if T >= 2:
        vel[1:T] = pos[1:T] - pos[:T-1]
    if T >= 3:
        acc[2:T] = vel[2:T] - vel[1:T-1]

    def L(a, b):
        try:
            return np.linalg.norm(pose_seq[:, a] - pose_seq[:, b], axis=1)
        except Exception:
            return np.zeros((T,), dtype=np.float32)

    limb = np.stack([
        L(23, 25),
        L(24, 26),
        L(11, 13),
        L(12, 14)
    ], axis=1).astype(np.float32)

    try:
        left = pose_seq[:, [11, 13, 15, 23, 25]]
        right = pose_seq[:, [12, 14, 16, 24, 26]]
        sym = np.linalg.norm(left - right, axis=2).astype(np.float32)
    except Exception:
        sym = np.zeros((T, 5), dtype=np.float32)

    feat = np.concatenate([pos, vel, acc, limb, sym], axis=1).astype(np.float32)
    return feat


# ----------------------------------------------------------
# Pose extraction
# ----------------------------------------------------------
def extract_poses_from_frames(frames, max_frames=120, stride=1):
    if len(frames) == 0:
        return None

    pose = mp_pose.Pose(static_image_mode=False)
    seq = []

    for f in frames[::stride][:max_frames]:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            coords = np.zeros((33, 3), dtype=np.float32)
        seq.append(coords)

    pose.close()

    if len(seq) == 0:
        return None

    return np.stack(seq)


def extract_pose_from_frame(frame):
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return np.zeros((33, 3), dtype=np.float32)


# ----------------------------------------------------------
# Models
# ----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, fdim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(fdim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ----------------------------------------------------------
# Model loading
# ----------------------------------------------------------
def load_model(model_path, model_type, fdim, device):
    if model_type == "lstm":
        model = LSTMNet(fdim)
    else:
        model = MLP(2 * fdim)

    state = torch.load(model_path, map_location="cpu")

    if "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
    except:
        # strip module. prefix if needed
        fixed = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(fixed)

    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------
# Load frames from file
# ----------------------------------------------------------
def load_frames(path, max_frames=120):
    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(path)
        frames = []
        ok, frame = cap.read()
        while ok and len(frames) < max_frames:
            frames.append(frame.copy())
            ok, frame = cap.read()
        cap.release()
        return frames

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Cannot read input file: " + path)
    return [img]


def pick_input_file(demos_dir="demos"):
    vids = []
    if os.path.isdir(demos_dir):
        vids = sorted(
            [
                f for f in os.listdir(demos_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".jpg", ".png"))
            ]
        )

    if vids:
        print("Pick a demo file:")
        for i, v in enumerate(vids):
            print(f"[{i}] {v}")
    else:
        print("No files found in demos/.")

    prompt = "Enter index, full file path, or press Enter for [0]: "
    while True:
        choice = input(prompt).strip()

        if choice == "" and vids:
            return os.path.join(demos_dir, vids[0])

        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(vids):
                return os.path.join(demos_dir, vids[idx])
            print("Index out of range. Try again.")
            continue

        if os.path.isfile(choice):
            return choice

        print("Invalid choice. Enter a valid index or a full file path.")


def pick_input_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(
            "Tkinter file dialog is unavailable on this Python install."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select video or image for deepfake detection",
        filetypes=[
            ("Video/Image files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not path:
        raise RuntimeError("No file selected.")
    return path


# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------
def run_inference(frames, model, model_type, device, max_frames=120, stride=1, out_path=None, fake_threshold=0.35):
    if len(frames) == 1:
        pose_seq = np.stack([extract_pose_from_frame(frames[0])])
    else:
        pose_seq = extract_poses_from_frames(frames, max_frames=max_frames, stride=stride)
        if pose_seq is None:
            raise RuntimeError("No poses detected.")

    feat_seq = compute_frame_features(pose_seq)  # (T, F)

    x = torch.from_numpy(feat_seq)[None, :, :].to(device)  # (1, T, F)
    lengths = torch.tensor([feat_seq.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        if model_type == "lstm":
            logits = model(x, lengths)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            inp = torch.cat([mean, std], dim=1)
            logits = model(inp)

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        real, fake = float(probs[0]), float(probs[1])

        # Use a lower threshold for fake to compensate for class imbalance bias
        pred = "FAKE" if fake >= fake_threshold else "REAL"
        conf = fake if pred == "FAKE" else real

    # write annotated video
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))

        for i, f in enumerate(frames[:max_frames:stride]):
            cv2.putText(f, f"{pred} ({conf:.3f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(f)
        writer.release()

    return pred, real, fake, conf


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", choices=["mlp", "lstm"], default="mlp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fake-threshold", type=float, default=0.35,
                    help="Classify as FAKE if fake probability >= this value (default: 0.35). "
                         "Lower = more sensitive to fakes.")
    ap.add_argument(
        "--file-dialog",
        action="store_true",
        help="Open OS file picker to choose input file.",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # If no input -> interactive selector or file dialog
    if not args.input:
        if args.file_dialog:
            args.input = pick_input_file_dialog()
        else:
            args.input = pick_input_file("demos")

    if not os.path.isfile(args.input):
        raise RuntimeError("Input file not found: " + str(args.input))

    # determine feature dim
    frames_tmp = load_frames(args.input, max_frames=1)
    pose_tmp = extract_pose_from_frame(frames_tmp[0])
    tmp_feat = compute_frame_features(np.stack([pose_tmp]))
    F = tmp_feat.shape[1]

    # load model
    model = load_model(args.model, args.model_type, F, device)

    # load frames
    frames = load_frames(args.input, max_frames=args.max_frames)

    t0 = time.time()
    pred, r, f, conf = run_inference(
        frames, model, args.model_type, device,
        max_frames=args.max_frames, stride=args.stride, out_path=args.out,
        fake_threshold=args.fake_threshold
    )
    t1 = time.time()

    print(f"Prediction: {pred}")
    print(f"Score REAL={r:.4f}, FAKE={f:.4f}")
    print(f"Confidence: {conf:.4f}")
    print(f"Inference time: {t1 - t0:.2f} sec")

    if args.out:
        print("Annotated video written to:", args.out)


if __name__ == "__main__":
    main()
