import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


class LSTMNet(torch.nn.Module):
    def __init__(self, fdim, hidden=128):
        super().__init__()
        self.lstm = torch.nn.LSTM(fdim, hidden, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_samples(features_root):
    xs, ys = [], []
    skipped = 0
    base = Path(features_root)
    for cls_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        label = 0 if cls_dir.name.lower().startswith("real") else 1
        for f in sorted(cls_dir.glob("*.npz")):
            npz = np.load(f, allow_pickle=True)
            feat = npz["features"].astype(np.float32)
            # Skip pre-aggregated legacy rows (1D). This evaluator expects per-frame (T,F).
            if feat.ndim == 1:
                skipped += 1
                continue
            if feat.ndim != 2:
                skipped += 1
                continue
            xs.append(feat)
            ys.append(label)
    if not xs:
        raise RuntimeError("No valid features found under " + str(base))
    if skipped:
        print(f"Skipped {skipped} non-sequence feature files.")
    return xs, np.array(ys, dtype=np.int64)


def pad_sequences(xs):
    max_t = max(x.shape[0] for x in xs)
    max_f = max(x.shape[1] for x in xs)
    out = np.zeros((len(xs), max_t, max_f), dtype=np.float32)
    lengths = np.zeros((len(xs),), dtype=np.int64)
    for i, x in enumerate(xs):
        t, f = x.shape
        out[i, :t, :f] = x
        lengths[i] = t
    return out, lengths, max_f


def aggregate_for_mlp(x):
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    return torch.cat([mean, std], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model-type", choices=["lstm", "mlp"], default="mlp")
    args = ap.parse_args()

    xs, y = load_samples(args.features)
    X, lengths, fdim = pad_sequences(xs)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    if args.model_type == "lstm":
        model = LSTMNet(fdim)
    else:
        model = MLP(2 * fdim)

    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except Exception:
        fixed = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(fixed)
    model.eval()

    with torch.no_grad():
        if args.model_type == "lstm":
            out = model(X_tensor, lengths_tensor)
        else:
            out = model(aggregate_for_mlp(X_tensor))
        probs = torch.softmax(out, dim=1)[:, 1].numpy()
        preds = out.argmax(dim=1).numpy()

    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")

    print(f"Samples: {len(y)}  FeatureDim: {fdim}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
