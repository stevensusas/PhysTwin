#!/usr/bin/env python3
"""Generate per-view GIFs for every case in vid2sim_dataset."""

import os
import re
import numpy as np
import cv2
import imageio

DATASET_DIR = os.path.join(os.path.dirname(__file__), "vid2sim_dataset")
FPS = 12


def load_view_frames(case_dir: str, view: int):
    data_dir = os.path.join(case_dir, "data")
    if not os.path.isdir(data_dir):
        return []
    pattern = re.compile(rf"^m_{view}_(\d+)\.png$")
    entries = []
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(data_dir, fname)))
    entries.sort(key=lambda x: x[0])
    frames = []
    for _, path in entries:
        img = cv2.imread(path)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def get_available_views(case_dir: str):
    data_dir = os.path.join(case_dir, "data")
    if not os.path.isdir(data_dir):
        return []
    views = set()
    for fname in os.listdir(data_dir):
        m = re.match(r"^m_(\d+)_\d+\.png$", fname)
        if m:
            views.add(int(m.group(1)))
    return sorted(views)


def tight_crop_bbox(frames, pad=0.12):
    """Compute stable tight bbox across all frames around non-white pixels."""
    y_min, y_max, x_min, x_max = 1e9, 0, 1e9, 0
    for frame in frames:
        mask = (frame < 240).any(axis=2)
        ys, xs = np.where(mask)
        if len(ys):
            y_min = min(y_min, ys.min())
            y_max = max(y_max, ys.max())
            x_min = min(x_min, xs.min())
            x_max = max(x_max, xs.max())
    if y_max == 0:
        h, w = frames[0].shape[:2]
        return 0, h, 0, w
    h, w = frames[0].shape[:2]
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    half = max(y_max - y_min, x_max - x_min) // 2
    half = int(half * (1 + pad))
    side = max(half * 2, 40)
    y0 = max(0, int(cy) - side // 2)
    y1 = min(h, y0 + side)
    x0 = max(0, int(cx) - side // 2)
    x1 = min(w, x0 + side)
    return y0, y1, x0, x1


def make_gif(frames, out_path):
    y0, y1, x0, x1 = tight_crop_bbox(frames)
    gif_frames = []
    for frame in frames:
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        padded = np.full((side, side, 3), 255, dtype=np.uint8)
        py = (side - ch) // 2
        px = (side - cw) // 2
        padded[py:py+ch, px:px+cw] = crop
        gif_frames.append(padded)
    imageio.mimsave(out_path, gif_frames, fps=FPS, loop=0)


def main():
    cases = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    print(f"Found {len(cases)} cases")

    for case in cases:
        case_dir = os.path.join(DATASET_DIR, case)
        views = get_available_views(case_dir)
        if not views:
            print(f"  {case}: no data, skipping")
            continue

        print(f"  {case}: {len(views)} views", end="", flush=True)
        for view in views:
            frames = load_view_frames(case_dir, view)
            if not frames:
                continue
            out_path = os.path.join(case_dir, f"replay_v{view}.gif")
            make_gif(frames, out_path)
            print(f" v{view}", end="", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
