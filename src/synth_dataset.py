# coding: utf-8
import argparse, os, numpy as np

def perturb(x):
    y = x.copy()
    y += np.random.normal(scale=0.08, size=y.shape)
    y[:,0] *= -1 if np.random.rand() < 0.3 else 1
    return y.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = []
    for sub in os.listdir(args.pose_folder):
        d = os.path.join(args.pose_folder, sub)
        if os.path.isdir(d):
            files += [os.path.join(d,f) for f in os.listdir(d) if f.endswith(".npz")]

    for i in range(args.num_samples):
        src = np.load(np.random.choice(files))["pose"]
        fake = perturb(src)
        np.savez_compressed(os.path.join(args.out,f"fake_{i}.npz"), pose=fake)

if __name__ == "__main__":
    main()
