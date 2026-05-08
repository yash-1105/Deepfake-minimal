#!/usr/bin/env python3
"""
Extract per-frame features from pose .npz files and save as per-sample .npz files
Usage:
python src/extract_features.py --pose-folder models/poses --out models/features_surreal --max-t 120 --stride 1
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ----------------------------------------------------------
# Load pose npz and force shape (T, 33, 3)
# ----------------------------------------------------------
def load_pose_npz(path):
    """
    Load pose sequence.
    Guarantee output shape: (T, 33, 3)
    Missing joints padded with zeros, extra joints truncated.
    """
    try:
        data = np.load(path, allow_pickle=True)

        pose = None
        for k in data.keys():
            arr = data[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3:
                pose = arr.astype(np.float32)
                break

        if pose is None:
            return None

        T, J, C = pose.shape
        if C != 3:
            return None

        # pad or truncate to exactly 33 joints
        if J < 33:
            pad = np.zeros((T, 33 - J, 3), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=1)
        elif J > 33:
            pose = pose[:, :33, :]

        return pose

    except Exception:
        return None


# ----------------------------------------------------------
# Compute per-frame features: (T, F)
# ----------------------------------------------------------
def compute_frame_features(pose_seq):
    """
    pose_seq: (T, 33, 3)
    returns: (T, F) float32
    Features = pos + vel + acc + limb(4) + symmetry(5)
    """
    T = pose_seq.shape[0]
    J = pose_seq.shape[1]

    # positions
    pos = pose_seq.reshape(T, -1)  # (T, 99)

    # velocity & acceleration
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    if T >= 2:
        vel[1:T] = pos[1:T] - pos[:T-1]

    if T >= 3:
        acc[2:T] = vel[2:T] - vel[1:T-1]

    # limb lengths
    def L(a, b):
        try:
            return np.linalg.norm(pose_seq[:, a] - pose_seq[:, b], axis=1)
        except Exception:
            return np.zeros((T,), dtype=np.float32)

    limb = np.stack([
        L(23,25),  # left leg
        L(24,26),  # right leg
        L(11,13),  # left arm
        L(12,14)   # right arm
    ], axis=1).astype(np.float32)

    # symmetry distances
    try:
        left = pose_seq[:, [11,13,15,23,25]]
        right = pose_seq[:, [12,14,16,24,26]]
        sym = np.linalg.norm(left - right, axis=2).astype(np.float32)  # (T,5)
    except Exception:
        sym = np.zeros((T,5), dtype=np.float32)

    # final vector per frame
    feat = np.concatenate([pos, vel, acc, limb, sym], axis=1).astype(np.float32)
    return feat  # (T, F)


# ----------------------------------------------------------
# Process class folder (real or fake)
# ----------------------------------------------------------
def process_folder(pose_folder, out_folder, cls_name, max_t=120, stride=1):
    src = Path(pose_folder) / cls_name
    dst = Path(out_folder) / cls_name
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix == '.npz'])

    for p in tqdm(files, desc=f"Processing {cls_name}", unit='file'):
        pose_seq = load_pose_npz(p)
        if pose_seq is None or pose_seq.shape[0] == 0:
            continue

        # stride + truncate
        pose_seq = pose_seq[::stride][:max_t]
        if pose_seq.shape[0] == 0:
            continue

        feat_seq = compute_frame_features(pose_seq)  # (T, F)

        out_path = dst / (p.stem + '.npz')
        label = 0 if cls_name.lower().startswith('real') else 1
        np.savez_compressed(str(out_path), features=feat_seq, label=int(label))

    return


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-t", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    pf = Path(args.pose_folder)
    if not pf.exists():
        raise RuntimeError("Pose folder not found: " + str(pf))

    subdirs = [p.name for p in pf.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError("Expected subfolders 'real' and 'fake' inside pose-folder.")

    for cls in subdirs:
        process_folder(args.pose_folder, args.out, cls, max_t=args.max_t, stride=args.stride)

    print("Feature extraction completed.")


if __name__ == "__main__":
    main()
