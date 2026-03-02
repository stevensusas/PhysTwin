#!/usr/bin/env python3
"""Render RGB video from saved synthetic fall data using Gaussian Splatting."""
import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage2"))

from qqtt.utils import cfg
cfg.load_from_yaml("configs/real.yaml")

from stage2.script_generate_fall_synthetic import render_gaussian_video, load_camera_meta

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--gs_model_path", type=str, required=True)
    parser.add_argument("--gs_source_path", type=str, required=True)
    parser.add_argument("--gs_iteration", type=int, default=10000)
    parser.add_argument("--gs_view_idx", type=int, default=0)
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--case_name", type=str, default=None)
    args = parser.parse_args()

    if args.base_path and args.case_name:
        load_camera_meta(args.base_path, args.case_name)

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    object_points = data["object_points"]
    out_dir = os.path.dirname(args.pkl_path)
    video_path = os.path.join(out_dir, "synthetic_fall_rgb.mp4")

    print(f"Rendering {len(object_points)} frames to {video_path}")
    render_gaussian_video(
        gs_model_path=args.gs_model_path,
        gs_source_path=args.gs_source_path,
        out_video_path=video_path,
        object_points=object_points,
        iteration=args.gs_iteration,
        view_idx=args.gs_view_idx,
        white_background=True,
    )
    print("Done!")
