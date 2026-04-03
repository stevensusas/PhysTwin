#!/usr/bin/env python3
"""
Generate Vid2Sim-format data from PhysTwin digital twins.

For each PhysTwin object, this script:
1) Simulates a free-fall using SpringMass.
2) Renders the sequence from Vid2Sim's 12 camera views using Gaussian Splatting.
3) Outputs in Vid2Sim's GSO dataset format (m_/a_/r_ images, transforms JSON, points3d.ply).
4) Optionally outputs Objaverse-style training format (single-view 16-frame renderings).
"""

import gc
import json
import math
import os
import pickle
import sys
from argparse import ArgumentParser
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import warp as wp

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage2"))

from qqtt.utils import cfg, logger

from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.gaussian_renderer import GaussianModel, render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.dynamic_utils import (
    get_topk_indices,
    interpolate_motions,
)
from gaussian_splatting.utils.graphics_utils import focal2fov

from stage2.script_generate_fall_synthetic import (
    load_reference_data,
    simulate_fall_springmass,
    simulate_interaction_springmass,
)

# Vid2Sim reference camera FOV (45 degrees)
VID2SIM_FOV = 0.7853981633974483  # radians
VID2SIM_RENDER_SIZE = 448

# GSO objects (e.g. bus) fill roughly [-1.3, 1.3]^3 in Vid2Sim's canonical frame.
# We normalize PhysTwin data to this scale so gso.yaml's floor_level=-0.7 works.
GSO_TARGET_HALF_EXTENT = 1.3


def load_vid2sim_camera_transforms(transforms_path: str) -> List[np.ndarray]:
    """Load the 12 c2w matrices from a Vid2Sim transforms_simulation.json."""
    with open(transforms_path, "r") as f:
        data = json.load(f)
    c2ws = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2ws.append(c2w)
    return c2ws


def compute_vid2sim_reference_center_and_scale(c2ws: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Compute the center point that all Vid2Sim cameras look at, and avg camera distance."""
    positions = np.array([c2w[:3, 3] for c2w in c2ws])
    # Vid2Sim cameras look at roughly the origin with some vertical offset
    # The average camera distance gives us the reference scale
    avg_distance = np.mean(np.linalg.norm(positions, axis=1))
    center = np.array([0.0, 0.0, -0.2])  # Vid2Sim objects are centered near (0,0,-0.2)
    return center, avg_distance


def create_vid2sim_cameras(
    object_center: np.ndarray,
    object_extent: float,
    gs_source_path: str,
    vid2sim_c2ws: List[np.ndarray],
    vid2sim_center: np.ndarray,
    vid2sim_avg_dist: float,
    render_size: int = VID2SIM_RENDER_SIZE,
    camera_zoom: float = 0.65,
) -> List[Tuple[Camera, int]]:
    """Create GS Camera objects matching Vid2Sim's 12 camera layout, scaled for the PhysTwin object.

    Args:
        object_center: center of the PhysTwin object in world coordinates.
        object_extent: approximate extent (diameter) of the object bounding box.
        gs_source_path: path to GS data dir (for loading an intrinsic template).
        vid2sim_c2ws: 12 c2w matrices from Vid2Sim reference.
        vid2sim_center: center point Vid2Sim cameras orbit around.
        vid2sim_avg_dist: average camera distance in Vid2Sim space.
        render_size: output image resolution (square).

    Returns:
        List of (Camera, view_index) tuples.
    """
    # Scale factor: map Vid2Sim camera distances to PhysTwin object scale
    # Vid2Sim objects are ~1 unit extent, cameras at ~1.8m
    # We scale camera positions proportionally to the PhysTwin object size
    scale = max(object_extent / 1.0, 0.1)

    fov_x = VID2SIM_FOV
    fov_y = VID2SIM_FOV  # square images

    cameras = []
    for view_idx, c2w_ref in enumerate(vid2sim_c2ws):
        # Extract camera position from the Vid2Sim OpenGL c2w and rescale
        cam_pos_ref = c2w_ref[:3, 3].copy()
        cam_pos_centered = cam_pos_ref - vid2sim_center
        cam_pos_scaled = cam_pos_centered * scale
        cam_pos_world = cam_pos_scaled + object_center

        # In OpenGL c2w, camera looks along -column2, up = column1
        # Both Vid2Sim and PhysTwin use Z-up world, so directions transfer directly
        forward_ref = -c2w_ref[:3, 2]
        forward = forward_ref / np.linalg.norm(forward_ref)

        # Build c2w in PhysTwin's GS convention:
        # Column 0 = right, Column 1 = up, Column 2 = forward (+Z = look direction)
        # This matches create_orbital_cameras() in script_generate_fall_synthetic.py
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos_world

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # stored transposed for GS
        T = w2c[:3, 3]

        cam = Camera(
            resolution=(render_size, render_size),
            colmap_id=view_idx,
            R=R,
            T=T,
            FoVx=fov_x,
            FoVy=fov_y,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name=f"vid2sim_view_{view_idx}",
            uid=view_idx,
            K=None,
        )
        cameras.append((cam, view_idx))

    return cameras


def precompute_gaussian_frames(
    gaussians: GaussianModel,
    object_points: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute Gaussian positions/rotations for all frames using LBS interpolation."""
    ctrl_pts = torch.tensor(object_points, dtype=torch.float32, device="cuda")
    n_steps = ctrl_pts.shape[0]

    xyz_0 = gaussians.get_xyz
    rgb_0 = gaussians.get_features_dc.squeeze(1)
    quat_0 = gaussians.get_rotation
    opa_0 = gaussians.get_opacity

    relations = get_topk_indices(ctrl_pts[0], K=16)
    all_pos = xyz_0.detach().clone()
    all_rot = quat_0.detach().clone()

    xyz = xyz_0.cpu()[None].repeat(n_steps, 1, 1)
    rgb = rgb_0.cpu()[None].repeat(n_steps, 1, 1)
    quat = quat_0.cpu()[None].repeat(n_steps, 1, 1)
    opa = opa_0.cpu()[None].repeat(n_steps, 1, 1)

    chunk_size = 20_000
    for i in range(1, n_steps):
        prev_particle_pos = ctrl_pts[i - 1]
        cur_particle_pos = ctrl_pts[i]
        motions = cur_particle_pos - prev_particle_pos

        num_chunks = (len(all_pos) + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, len(all_pos))
            all_pos_chunk = all_pos[start:end]
            all_rot_chunk = all_rot[start:end]
            all_pos_chunk, all_rot_chunk, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=motions,
                relations=relations,
                xyz=all_pos_chunk,
                quat=all_rot_chunk,
                device="cuda",
                step=str(i),
            )
            all_pos[start:end] = all_pos_chunk
            all_rot[start:end] = all_rot_chunk

        xyz[i] = all_pos.detach().cpu()
        quat[i] = all_rot.detach().cpu()
        rgb[i] = rgb[i - 1]
        opa[i] = opa[i - 1]

    quat = torch.nn.functional.normalize(quat, dim=-1)
    return xyz, rgb, quat, opa


def render_vid2sim_frames(
    gs_model_path: str,
    gs_source_path: str,
    output_dir: str,
    object_points: np.ndarray,
    cameras: List[Tuple[Camera, int]],
    num_output_frames: Optional[int] = None,
    iteration: int = -1,
    also_render_training: bool = True,
    object_center: Optional[np.ndarray] = None,
) -> None:
    """Render all frames for all views and save in Vid2Sim format.

    Saves:
      {output_dir}/data/m_{view}_{frame}.png  (RGB, white bg)
      {output_dir}/data/a_{view}_{frame}.png  (RGBA with alpha)
      {output_dir}/data/r_{view}_{frame}.png  (same as m_ for white bg)
      {output_dir}/data/r_{view}_-1.png       (pure white background)
      {output_dir}/renderings/{frame:03d}.png  (training format, front view only)
    """
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    gs_args = parser.parse_args([])
    gs_args.source_path = os.path.abspath(gs_source_path)
    gs_args.model_path = gs_model_path
    gs_args.white_background = True
    gs_args.data_device = "cuda"
    gs_args.gs_init_opt = "hybrid"
    gs_args.use_masks = True
    gs_args.pts_per_triangles = 30
    gs_args.use_high_res = False
    gs_args.train_test_exp = False
    gs_args.images = "images"
    gs_args.depths = ""
    gs_args.eval = False
    gs_args.sh_degree = 3
    gs_args.resolution = 1
    gs_args.isotropic = True
    gs_args.disable_sh = False

    dataset = model.extract(gs_args)
    pipe = pipeline_params.extract(gs_args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene_obj = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Shift Gaussians to origin so they align with LGM's canonical coordinate frame.
    # LGM generates pred.ply centered at (0,0,0); cameras in transforms_train.json
    # must look at the same origin, so everything must be in this centered frame.
    if object_center is not None and np.linalg.norm(object_center) > 1e-6:
        oc = torch.tensor(object_center, dtype=torch.float32, device="cuda")
        gaussians._xyz = gaussians._xyz - oc
        logger.info(f"[Vid2Sim] Shifted Gaussians by -object_center {object_center.round(4).tolist()}")

    logger.info("[Vid2Sim] Precomputing Gaussian frames...")
    xyz, rgb, quat, opa = precompute_gaussian_frames(gaussians, object_points)

    n_sim_frames = xyz.shape[0]
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Select frame indices to output (subsample simulation frames to num_output_frames)
    if num_output_frames is None or num_output_frames >= n_sim_frames:
        frame_indices = np.arange(n_sim_frames)
    else:
        frame_indices = np.linspace(0, n_sim_frames - 1, num_output_frames, dtype=int)

    logger.info(f"[Vid2Sim] Rendering {len(cameras)} views x {len(frame_indices)} frames...")

    for cam, view_idx in cameras:
        # Save white background image (no object)
        white_bg = np.ones((cam.image_height, cam.image_width, 4), dtype=np.uint8) * 255
        cv2.imwrite(
            os.path.join(data_dir, f"r_{view_idx}_-1.png"),
            cv2.cvtColor(white_bg, cv2.COLOR_RGBA2BGRA),
        )

        for out_frame_idx, sim_frame_idx in enumerate(frame_indices):
            # Set Gaussian state for this frame
            gaussians._xyz = xyz[sim_frame_idx].to("cuda")
            gaussians._features_dc = rgb[sim_frame_idx].unsqueeze(1).to("cuda")
            gaussians._rotation = quat[sim_frame_idx].to("cuda")
            gaussians._opacity = gaussians.inverse_opacity_activation(
                opa[sim_frame_idx]
            ).to("cuda")

            results = render(
                cam, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
            )
            rendered = results["render"].detach().cpu()

            # render_gsplat returns (4,H,W) with alpha, render_3dgs returns (3,H,W)
            if rendered.shape[0] == 4:
                rendering = rendered[:3]  # (3, H, W)
                alpha = rendered[3:4]     # (1, H, W)
            else:
                rendering = rendered[:3]
                # Estimate alpha: pixels that differ from white background
                alpha = 1.0 - (rendering.min(dim=0, keepdim=True).values > 0.99).float()

            # Convert to numpy uint8
            rgb_np = (rendering.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            alpha_np = (alpha.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)

            # m_{view}_{frame}.png - RGB on white background (saved as RGB)
            m_path = os.path.join(data_dir, f"m_{view_idx}_{out_frame_idx}.png")
            cv2.imwrite(m_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            # a_{view}_{frame}.png - RGBA with alpha
            rgba_np = np.concatenate([rgb_np, alpha_np[:, :, None]], axis=2)
            a_path = os.path.join(data_dir, f"a_{view_idx}_{out_frame_idx}.png")
            cv2.imwrite(a_path, cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA))

            # r_{view}_{frame}.png - RGB on white background, fully opaque (no alpha)
            r_path = os.path.join(data_dir, f"r_{view_idx}_{out_frame_idx}.png")
            cv2.imwrite(r_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    logger.info(f"[Vid2Sim] Saved {len(cameras) * len(frame_indices)} frame images to {data_dir}")

    # Training format: single front view (view 0), 16 frames
    if also_render_training:
        train_dir = os.path.join(output_dir, "renderings")
        os.makedirs(train_dir, exist_ok=True)
        n_train = 16
        if n_sim_frames >= n_train:
            train_indices = np.linspace(0, n_sim_frames - 1, n_train, dtype=int)
        else:
            train_indices = np.arange(n_sim_frames)

        front_cam = cameras[0][0]  # view 0 is the front view
        for out_idx, sim_idx in enumerate(train_indices):
            gaussians._xyz = xyz[sim_idx].to("cuda")
            gaussians._features_dc = rgb[sim_idx].unsqueeze(1).to("cuda")
            gaussians._rotation = quat[sim_idx].to("cuda")
            gaussians._opacity = gaussians.inverse_opacity_activation(
                opa[sim_idx]
            ).to("cuda")

            results = render(
                front_cam, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
            )
            rendering = results["render"][:3].detach().cpu()
            rgb_np = (rendering.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            train_path = os.path.join(train_dir, f"{out_idx:03d}.png")
            cv2.imwrite(train_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        logger.info(f"[Vid2Sim] Saved {len(train_indices)} training frames to {train_dir}")


def generate_transforms_json(
    cameras: List[Tuple[Camera, int]],
    output_path: str,
    norm_scale: float = 1.0,
) -> None:
    """Write a transforms_simulation.json matching Vid2Sim format.

    Args:
        norm_scale: uniform scale applied to camera positions so the output
            geometry matches GSO conventions (floor_level=-0.7 etc.).
    """
    frames = []
    for cam, view_idx in cameras:
        # Recover c2w from the Camera object (PhysTwin GS convention)
        R_stored = np.array(cam.R)  # transposed w2c rotation
        T_stored = np.array(cam.T)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_stored.T  # un-transpose to get w2c rotation
        w2c[:3, 3] = T_stored
        c2w_gs = np.linalg.inv(w2c)

        # Convert from PhysTwin GS convention (right/up/forward) to
        # OpenGL/Blender convention (right/up/-back) for Vid2Sim JSON:
        # GS: col0=right, col1=up, col2=forward
        # OpenGL: col0=right, col1=up, col2=back (negate col2)
        c2w = c2w_gs.copy()
        c2w[:3, 2] *= -1  # negate forward to get back

        # Scale camera position to match GSO coordinate scale
        c2w[:3, 3] *= norm_scale

        frames.append({
            "file_path": f"./data/m_{view_idx}_0",
            "time": 0.0,
            "rotation": 0.0,
            "transform_matrix": c2w.tolist(),
        })

    data = {
        "camera_angle_x": VID2SIM_FOV,
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_points3d_ply(points: np.ndarray, output_path: str) -> None:
    """Export object points as a PLY file."""
    n = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(output_path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def main():
    parser = ArgumentParser(description="Generate Vid2Sim-format data from PhysTwin digital twins")
    parser.add_argument("--data_path", type=str, required=True, help="Path to final_data.pkl")
    parser.add_argument("--case_name", type=str, required=True, help="PhysTwin case name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Vid2Sim data")
    parser.add_argument("--gs_model_path", type=str, required=True, help="Path to GS model")
    parser.add_argument("--gs_source_path", type=str, required=True, help="Path to GS source data")
    parser.add_argument("--gs_iteration", type=int, default=10000)
    parser.add_argument("--springmass_checkpoint", type=str, default=None,
                        help="Optional checkpoint for spring-mass parameters")
    parser.add_argument("--vid2sim_transforms", type=str,
                        default=None,
                        help="Path to a Vid2Sim transforms_simulation.json for camera layout reference")
    parser.add_argument("--num_sim_frames", type=int, default=24, help="Number of simulation frames")
    parser.add_argument("--num_output_frames", type=int, default=None, help="Number of output frames per view (default: all simulation frames)")
    parser.add_argument("--fall_height", type=float, default=0.15)
    parser.add_argument("--camera_zoom", type=float, default=0.65,
                        help="Camera distance scale; < 1 = more zoomed in (default 0.65)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render_size", type=int, default=VID2SIM_RENDER_SIZE)
    parser.add_argument("--no_training_format", action="store_true",
                        help="Skip generating Objaverse-style training data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.set_log_file(path=args.output_dir, name="vid2sim_gen_log")
    cfg.device = args.device

    # --- Load Vid2Sim reference cameras ---
    transforms_path = args.vid2sim_transforms
    if transforms_path is None:
        # Default: use bus transforms as reference
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "Vid2Sim", "dataset", "bus", "transforms_simulation.json"
        )
        if os.path.exists(default_path):
            transforms_path = default_path
        else:
            raise FileNotFoundError(
                "No Vid2Sim transforms_simulation.json found. "
                "Provide --vid2sim_transforms or ensure ../Vid2Sim/dataset/bus/transforms_simulation.json exists."
            )

    vid2sim_c2ws = load_vid2sim_camera_transforms(transforms_path)
    vid2sim_center, vid2sim_avg_dist = compute_vid2sim_reference_center_and_scale(vid2sim_c2ws)
    logger.info(f"[Vid2Sim] Loaded {len(vid2sim_c2ws)} reference cameras (avg dist={vid2sim_avg_dist:.3f})")

    # --- Load config ---
    config_file = "configs/cloth.yaml" if "cloth" in args.case_name else "configs/real.yaml"
    cfg.load_from_yaml(config_file)
    cfg.device = args.device

    # --- Simulate interaction or free-fall ---
    logger.info(f"[Vid2Sim] Loading reference data from {args.data_path}")
    ref = load_reference_data(args.data_path, args.device)
    structure_points = ref["structure_points"]
    num_original_points = ref["num_original_points"]

    raw_controller_points = ref["raw"].get("controller_points", None)
    has_interaction = raw_controller_points is not None and raw_controller_points.shape[1] > 0

    if has_interaction:
        logger.info(
            f"[Vid2Sim] Replaying interaction ({raw_controller_points.shape[0]} frames, "
            f"N_ctrl={raw_controller_points.shape[1]})"
        )
        object_points_array, floor_height, floor_axis = simulate_interaction_springmass(
            points=structure_points,
            num_original_points=num_original_points,
            device=args.device,
            data=ref["raw"],
            controller_points=raw_controller_points.astype(np.float32),
            checkpoint_path=args.springmass_checkpoint,
            case_name=args.case_name,
        )
    else:
        logger.info(f"[Vid2Sim] Simulating free-fall ({args.num_sim_frames} frames, height={args.fall_height})")
        object_points_array, floor_height, floor_axis = simulate_fall_springmass(
            points=structure_points,
            num_original_points=num_original_points,
            num_frames=args.num_sim_frames,
            device=args.device,
            fall_height=args.fall_height,
            data=ref["raw"],
            checkpoint_path=args.springmass_checkpoint,
            case_name=args.case_name,
        )
    logger.info(f"[Vid2Sim] Simulation complete: {object_points_array.shape}")

    # --- Compute object center and extent for camera scaling ---
    # object_extent uses the full trajectory bounding box so cameras see the entire fall.
    # object_center uses frame 0 particle mean so that:
    #   1. Frame 0 sits at the origin after centering — matching LGM's canonical frame.
    #   2. floor_level encodes the full fall distance (frame 0 → floor), not just the
    #      mid-trajectory-to-floor distance that the all-frames bbox center would give.
    all_pts = object_points_array.reshape(-1, 3)  # (N*T, 3)
    obj_min = all_pts.min(axis=0)
    obj_max = all_pts.max(axis=0)
    object_extent = np.linalg.norm(obj_max - obj_min)
    object_center = object_points_array[0].mean(axis=0)  # frame 0 particle mean
    logger.info(f"[Vid2Sim] Object center (frame 0 mean)={object_center}, extent={object_extent:.4f}")

    # Normalize: shift the entire trajectory to origin so cameras (which are built
    # around origin) align with LGM's canonical coordinate frame for pred.ply.
    object_points_array = object_points_array - object_center
    first_frame_pts = object_points_array[0]
    logger.info(f"[Vid2Sim] Trajectory centered at frame 0 mean (was {object_center.round(4).tolist()})")

    # --- Compute normalization scale to match GSO conventions ---
    # GSO objects fill roughly [-1.3, 1.3]^3.  Scale our geometry so that
    # gso.yaml's floor_level=-0.7 and gravity settings work correctly.
    first_half_extents = np.abs(first_frame_pts).max(axis=0)
    max_half_extent = first_half_extents.max()
    norm_scale = GSO_TARGET_HALF_EXTENT / max(max_half_extent, 1e-6)
    logger.info(f"[Vid2Sim] norm_scale={norm_scale:.4f} (max_half_extent={max_half_extent:.4f})")

    # --- Export points3d.ply (at GSO scale) ---
    ply_path = os.path.join(args.output_dir, "points3d.ply")
    export_points3d_ply(first_frame_pts * norm_scale, ply_path)
    logger.info(f"[Vid2Sim] Exported {first_frame_pts.shape[0]} points to {ply_path}")

    # --- Export hand trajectory in Vid2Sim normalized coords ---
    # Apply same transform as object: subtract object_center, multiply by norm_scale.
    if has_interaction:
        ctrl_norm = (raw_controller_points.astype(np.float32) - object_center[None, None, :]) * norm_scale
        hand_traj_path = os.path.join(args.output_dir, "hand_trajectory.npy")
        np.save(hand_traj_path, ctrl_norm)
        logger.info(
            f"[Vid2Sim] Exported hand trajectory: {ctrl_norm.shape} to {hand_traj_path}"
        )
    else:
        logger.info("[Vid2Sim] No controller points found — skipping hand_trajectory.npy")

    # --- Free simulation memory ---
    n_nodes = len(structure_points)
    del ref, structure_points
    gc.collect()
    torch.cuda.empty_cache()
    wp.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # --- Create cameras ---
    # Cameras orbit around origin (object is now at origin after centering above).
    # This ensures transforms_train.json cameras look at (0,0,0), matching LGM canonical frame.
    cameras = create_vid2sim_cameras(
        object_center=np.zeros(3),
        object_extent=object_extent,
        gs_source_path=os.path.abspath(args.gs_source_path),
        vid2sim_c2ws=vid2sim_c2ws,
        vid2sim_center=vid2sim_center,
        vid2sim_avg_dist=vid2sim_avg_dist,
        render_size=args.render_size,
        camera_zoom=args.camera_zoom,
    )
    logger.info(f"[Vid2Sim] Created {len(cameras)} cameras")

    # --- Generate transforms JSONs (at GSO scale) ---
    for name in ["transforms_simulation", "transforms_train", "transforms_val", "transforms_test"]:
        json_path = os.path.join(args.output_dir, f"{name}.json")
        generate_transforms_json(cameras, json_path, norm_scale=norm_scale)
    logger.info(f"[Vid2Sim] Wrote transform JSON files (norm_scale={norm_scale:.4f})")

    # --- Write per-case simulation config for Vid2Sim ---
    # PhysTwin convention: objects fall toward +Z, floor is above (+Z side).
    # Vid2Sim/Kaolin convention: gravity force in -Z, floor is below (-Z side).
    # These are mirror images, so we negate the normalized floor level.
    floor_level_normalized = -abs((floor_height - object_center[floor_axis]) * norm_scale)
    # Scale gravity by node count relative to double_stretch_zebra (g_ref=9.8).
    # Heavier objects (more nodes) get more gravity to match GT fall speed.
    G_REF = 9.8
    _ref_data_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(args.data_path)),
        "..", "double_stretch_zebra", "final_data.pkl",
    ))
    if os.path.exists(_ref_data_path) and args.case_name != "double_stretch_zebra":
        _ref_data = load_reference_data(_ref_data_path, args.device)
        N_REF = len(_ref_data["structure_points"])
        del _ref_data
    else:
        N_REF = n_nodes  # fallback: when processing double_stretch_zebra itself
    gravity = float(G_REF * n_nodes / N_REF)
    sim_config = {
        "floor_level": float(floor_level_normalized),
        "floor_axis": int(floor_axis),
        "flip_floor": False,
        "gravity": gravity,
    }
    sim_config_path = os.path.join(args.output_dir, "sim_config.yaml")
    import yaml
    with open(sim_config_path, "w") as f:
        yaml.dump(sim_config, f, default_flow_style=False)
    logger.info(
        f"[Vid2Sim] Wrote sim_config.yaml: floor_level={floor_level_normalized:.4f}, "
        f"floor_axis={floor_axis}, flip_floor=False, gravity={gravity:.4f} (N={n_nodes})"
    )

    # --- Render frames ---
    render_vid2sim_frames(
        gs_model_path=args.gs_model_path,
        gs_source_path=args.gs_source_path,
        output_dir=args.output_dir,
        object_points=object_points_array,
        cameras=cameras,
        num_output_frames=args.num_output_frames,
        iteration=args.gs_iteration,
        also_render_training=not args.no_training_format,
        object_center=object_center,
    )

    logger.info(f"[Vid2Sim] Done! Output at {args.output_dir}")


if __name__ == "__main__":
    main()
