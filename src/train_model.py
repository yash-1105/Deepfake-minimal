#!/usr/bin/env python3
"""
Final, stable train_model.py — robust to mixed feature shapes (1D aggregated or 2D per-frame).
This version computes a GLOBAL feature width (max_f) across all files and pads every batch to that width,
ensuring the LSTM input_size matches actual tensor width.

Usage:
python src/train_model.py --features models/features_surreal --out models/model_surreal_lstm.pt --epochs 12 --batch 32 --model lstm
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# -------------------------
# Dataset
# -------------------------
class FeatureSeqDataset(Dataset):
    """
    Each .npz contains:
        features: (T, F)  OR  (F,) for aggregated
        label: int
    __getitem__ returns (np.ndarray (T,F)), label, T
    """

    def __init__(self, file_pairs):
        self.data = file_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        d = np.load(path, allow_pickle=True)
        x = d["features"]
        # normalize 1D -> (1, F)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise RuntimeError(f"Invalid feature shape in {path}: {x.shape}")
        return x.astype(np.float32), int(label), x.shape[0]


# -------------------------
# Collate factory — pads to GLOBAL max_f
# -------------------------
def make_collate_fn(global_max_f):
    def collate_fn(batch):
        """
        Pads sequences to (B, max_T, global_max_f).
        """
        xs, ys, lens = zip(*batch)
        max_t = max([x.shape[0] for x in xs])
        B = len(xs)
        padded = torch.zeros(B, max_t, global_max_f, dtype=torch.float32)
        for i, x in enumerate(xs):
            t, f = x.shape
            padded[i, :t, :f] = torch.from_numpy(x)
        return padded, torch.tensor(ys, dtype=torch.long), torch.tensor(lens, dtype=torch.long)
    return collate_fn


# -------------------------
# Models
# -------------------------
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
        # x: (B, D)
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, fdim, hidden=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=fdim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 2)

    def forward(self, x, lengths):
        # x: (B, T, F)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # h: (num_layers * num_dirs, B, hidden)
        if self.bidirectional:
            last_f = h[-2]
            last_b = h[-1]
            last = torch.cat([last_f, last_b], dim=1)
        else:
            last = h[-1]
        return self.fc(last)


# -------------------------
# Utilities
# -------------------------
def build_file_list(features_folder):
    base = Path(features_folder)
    pairs = []
    if not base.exists():
        raise RuntimeError("Features folder not found: " + str(base))
    for cls_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        label = 0 if cls_dir.name.lower().startswith("real") else 1
        for f in sorted(cls_dir.glob("*.npz")):
            pairs.append((str(f), label))
    return pairs


def compute_global_feature_dim(file_pairs):
    """
    Return maximum feature width across all files.
    """
    max_f = 0
    for path, _ in tqdm(file_pairs, desc="Scanning feature dims", ncols=80):
        try:
            d = np.load(path, allow_pickle=True)
            x = d["features"]
            if x.ndim == 1:
                f = x.shape[0]
            elif x.ndim == 2:
                f = x.shape[1]
            else:
                raise RuntimeError(f"Unexpected feature ndim in {path}: {x.shape}")
            if f > max_f:
                max_f = f
        except Exception as e:
            print("Warning: skipping", path, "->", e)
    if max_f <= 0:
        raise RuntimeError("Failed to determine feature dimension (max_f <= 0)")
    return int(max_f)


def aggregate_for_mlp(x):
    # x: (B, T, F)
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    return torch.cat([mean, std], dim=1)


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


# -------------------------
# Training / Eval loops
# -------------------------
def train_epoch(model, loader, opt, device, model_type, class_weights=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for xb, yb, lengths in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        lengths = lengths.to(device)

        opt.zero_grad()
        if model_type == "mlp":
            inp = aggregate_for_mlp(xb)  # (B, 2F)
            logits = model(inp)
        else:
            logits = model(xb, lengths)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        bsz = yb.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, yb) * bsz
        total += bsz

    return total_loss / total, total_acc / total


def eval_epoch(model, loader, device, model_type, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    with torch.no_grad():
        for xb, yb, lengths in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)
            if model_type == "mlp":
                inp = aggregate_for_mlp(xb)
                logits = model(inp)
            else:
                logits = model(xb, lengths)
            loss = criterion(logits, yb)
            bsz = yb.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy(logits, yb) * bsz
            total += bsz
    return total_loss / total, total_acc / total


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Features folder with class subfolders (real/, fake/)")
    ap.add_argument("--out", required=True, help="Output model path (.pt)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", choices=["mlp", "lstm"], default="mlp")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # build file list
    files = build_file_list(args.features)
    if len(files) == 0:
        raise RuntimeError("No feature files found under: " + args.features)

    # compute global feature width (max_f)
    max_f = compute_global_feature_dim(files)
    print("Global feature width (F) =", max_f)

    # stratified split
    labels = [l for _, l in files]
    idx_train, idx_val = train_test_split(range(len(files)), test_size=0.2, stratify=labels, random_state=args.seed)
    train_files = [files[i] for i in idx_train]
    val_files = [files[i] for i in idx_val]

    # Compute class weights to counteract imbalance (real >> fake)
    train_labels = [l for _, l in train_files]
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    n_total = n_real + n_fake
    # Weight inversely proportional to class frequency
    w_real = n_total / (2.0 * n_real) if n_real > 0 else 1.0
    w_fake = n_total / (2.0 * n_fake) if n_fake > 0 else 1.0
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    print(f"Class balance → real: {n_real}, fake: {n_fake}")
    print(f"Class weights  → real: {w_real:.3f}, fake: {w_fake:.3f}")

    # datasets
    train_ds = FeatureSeqDataset(train_files)
    val_ds = FeatureSeqDataset(val_files)

    # collate fixed to global max_f
    collate_fn = make_collate_fn(max_f)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # Build model using global feature width
    F = max_f
    if args.model == "mlp":
        model = MLP(2 * F)
    else:
        model = LSTMNet(F, hidden=128)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = -1.0
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device, args.model, class_weights=class_weights)
        val_loss, val_acc = eval_epoch(model, val_loader, device, args.model, class_weights=class_weights)
        print(f"epoch {ep} train_loss {tr_loss:.4f} train_acc {tr_acc:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out)
            print("Saved best model ->", args.out)

    print("Training finished. Best val acc:", best_val)


if __name__ == "__main__":
    main()
