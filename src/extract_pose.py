import os
import argparse
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import cv2

def extract_pose_from_frames(frame_dir, max_frames=120, stride=1):
    """Extract pose landmarks for a sequence of frames using MediaPipe."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    frames = sorted([os.path.join(frame_dir, f) 
                     for f in os.listdir(frame_dir)
                     if f.lower().endswith(('.jpg', '.png'))])

    frames = frames[::stride][:max_frames]  # apply stride + limit
    seq = []

    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            coords = np.zeros((33, 3), dtype=np.float32)
        
        seq.append(coords)

    pose.close()
    if len(seq) == 0:
        return None

    return np.stack(seq)   # shape: (T, 33, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to frames root")
    parser.add_argument("--output", required=True, help="Output folder for npz")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_real = os.path.join(args.output, "real")
    os.makedirs(out_real, exist_ok=True)

    # Each subfolder inside the Penn_Action/frames directory is one sequence
    for seq_name in tqdm(os.listdir(args.input)):
        seq_path = os.path.join(args.input, seq_name)
        if not os.path.isdir(seq_path):
            continue
        
        pose_seq = extract_pose_from_frames(
            seq_path,
            max_frames=args.max_frames,
            stride=args.stride
        )

        if pose_seq is None:
            continue

        # Save compressed
        out_file = os.path.join(out_real, f"{seq_name}.npz")
        np.savez_compressed(out_file, pose=pose_seq)

    print("Pose extraction completed.")


if __name__ == "__main__":
    main()
