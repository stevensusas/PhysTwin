#!/usr/bin/env python3
"""Batch runner: generate synthetic fall data + 4-view PC & RGB videos for all PhysTwin objects.

Usage:
    conda activate phystwin
    cd /path/to/PhysTwin
    python run_fall_synthetic_batch.py [--cases CASE ...]
"""

import glob
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage2"))

ALL_CASES = [
    "double_lift_cloth_1",
    "double_lift_cloth_3",
    "double_lift_sloth",
    "double_lift_zebra",
    "double_stretch_sloth",
    "double_stretch_zebra",
    "rope_double_hand",
    "single_clift_cloth_1",
    "single_clift_cloth_3",
    "single_lift_cloth",
    "single_lift_cloth_1",
    "single_lift_cloth_3",
    "single_lift_cloth_4",
    "single_lift_dinosor",
    "single_lift_rope",
    "single_lift_sloth",
    "single_lift_zebra",
    "single_push_rope",
    "single_push_rope_1",
    "single_push_rope_4",
    "single_push_sloth",
    "weird_package",
]

GS_MODEL_SUBDIR = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

# Default Vid2Sim reference transforms
VID2SIM_TRANSFORMS = os.path.join(
    os.path.dirname(__file__), "..", "Vid2Sim", "dataset", "bus", "transforms_simulation.json"
)


def find_best_checkpoint(case: str) -> str:
    pattern = f"experiments/{case}/train/best_*.pth"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No best_*.pth found for {case}")
    return matches[0]


def is_cloth(case: str) -> bool:
    return "cloth" in case


def run_case(case: str, vid2sim_transforms: str) -> None:
    from qqtt.utils import cfg

    config = "configs/cloth.yaml" if is_cloth(case) else "configs/real.yaml"
    cfg.load_from_yaml(config)

    data_path = f"data/different_types/{case}/final_data.pkl"
    checkpoint = find_best_checkpoint(case)
    gs_model = f"gaussian_output/{case}/{GS_MODEL_SUBDIR}"
    gs_source = f"data/gaussian_data/{case}"
    output_dir = f"outputs/fall_{case}"
    output_pkl = f"{output_dir}/synthetic_fall.pkl"

    for path in [data_path, checkpoint, gs_model, gs_source]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    os.makedirs(output_dir, exist_ok=True)

    sys.argv = [
        "run_fall_synthetic.py",
        "--data_path", data_path,
        "--simulator", "springmass",
        "--springmass_checkpoint", checkpoint,
        "--output_path", output_pkl,
        "--render_pointcloud",
        "--render_rgb",
        "--base_path", "data/different_types",
        "--case_name", case,
        "--gs_model_path", gs_model,
        "--gs_source_path", gs_source,
        "--gs_iteration", "10000",
        "--fall_height", "0.15",
        "--num_frames", "24",
        "--vid2sim_transforms", vid2sim_transforms,
    ]

    from stage2.script_generate_fall_synthetic import main
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Subset of cases to run (default: all)")
    parser.add_argument("--vid2sim_transforms", type=str, default=VID2SIM_TRANSFORMS,
                        help="Path to Vid2Sim reference transforms_simulation.json")
    args = parser.parse_args()

    cases = args.cases if args.cases else ALL_CASES

    if not os.path.exists(args.vid2sim_transforms):
        print(f"ERROR: Vid2Sim transforms not found at {args.vid2sim_transforms}")
        print("Please provide --vid2sim_transforms pointing to a valid transforms_simulation.json")
        sys.exit(1)

    results = {}
    t0 = time.time()
    for i, case in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cases)}] Processing: {case}")
        print(f"{'='*60}")
        case_t0 = time.time()
        try:
            run_case(case, args.vid2sim_transforms)
            elapsed = time.time() - case_t0
            results[case] = f"OK ({elapsed:.1f}s)"
            print(f"[{case}] Done in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - case_t0
            results[case] = f"FAILED ({elapsed:.1f}s): {e}"
            print(f"[{case}] FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
    for case, status in results.items():
        print(f"  {case}: {status}")
