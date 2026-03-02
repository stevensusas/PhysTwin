#!/usr/bin/env python3
"""
Generate synthetic fall sequences using a trained PhysTwin digital twin.

This script:
1) Loads a trained SimplicitsObject (digital twin) from pickle.
2) Simulates a free-fall onto a floor from a fixed height.
3) Saves data in RealData-compatible format.
4) Optionally renders an RGB video using Gaussian Splatting.
"""

import json
import math
import os
import pickle
from argparse import ArgumentParser
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree
from kaolin.physics.simplicits.easy_api import SimplicitsScene

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qqtt.utils import cfg, logger
from qqtt.model.diff_simulator import SpringMassSystemWarp

# Gaussian rendering (optional)
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.gaussian_renderer import GaussianModel, render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.dynamic_utils import get_topk_indices, interpolate_motions, knn_weights
from gaussian_splatting.utils.graphics_utils import focal2fov


def load_reference_data(data_path: str, device: str) -> Dict[str, Any]:
    """Load reference data used for structure points and metadata."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    required_keys = [
        "object_points",
        "object_colors",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in reference data: {missing}")

    object_points = torch.tensor(data["object_points"], dtype=torch.float32, device=device)
    surface_points = torch.tensor(data["surface_points"], dtype=torch.float32, device=device)
    interior_points = torch.tensor(data["interior_points"], dtype=torch.float32, device=device)

    structure_points = torch.cat([object_points[0], surface_points, interior_points], dim=0)
    num_original_points = object_points.shape[1]

    return {
        "raw": data,
        "object_points": object_points,
        "structure_points": structure_points,
        "num_original_points": num_original_points,
    }


def load_camera_meta(base_path: str, case_name: str) -> None:
    """Populate cfg with intrinsics and camera poses for visualize_pc."""
    import numpy as np
    import json

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def build_springmass_simulator(
    points: torch.Tensor,
    device: str,
    object_radius: float,
    object_max_neighbours: int,
    dt: float,
    num_substeps: int,
    spring_Y: float,
    collide_elas: float,
    collide_fric: float,
    dashpot_damping: float,
    drag_damping: float,
    collide_object_elas: float,
    collide_object_fric: float,
    collision_dist: float,
    reverse_z: bool,
    spring_Y_min: float,
    spring_Y_max: float,
    data: Dict[str, Any],
) -> SpringMassSystemWarp:
    """Create a SpringMassSystemWarp simulator for free-fall."""
    pts_np = points.detach().cpu().numpy()
    tree = cKDTree(pts_np)

    springs = []
    rest_lengths = []
    spring_flags = np.zeros((len(pts_np), len(pts_np)), dtype=np.uint8)

    for i in range(len(pts_np)):
        idx = tree.query_ball_point(pts_np[i], r=object_radius)
        if i in idx:
            idx.remove(i)
        if object_max_neighbours is not None and len(idx) > object_max_neighbours:
            idx = idx[:object_max_neighbours]
        for j in idx:
            if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                rest_length = np.linalg.norm(pts_np[i] - pts_np[j])
                if rest_length > 1e-4:
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    springs.append([i, j])
                    rest_lengths.append(rest_length)

    springs = torch.tensor(springs, dtype=torch.int32, device=device)
    rest_lengths = torch.tensor(rest_lengths, dtype=torch.float32, device=device)
    masses = torch.ones(len(pts_np), dtype=torch.float32, device=device)
    return SpringMassSystemWarp(
        init_vertices=points,
        init_springs=springs,
        init_rest_lengths=rest_lengths,
        init_masses=masses,
        dt=dt,
        num_substeps=num_substeps,
        spring_Y=spring_Y,
        collide_elas=collide_elas,
        collide_fric=collide_fric,
        dashpot_damping=dashpot_damping,
        drag_damping=drag_damping,
        collide_object_elas=collide_object_elas,
        collide_object_fric=collide_object_fric,
        collision_dist=collision_dist,
        num_object_points=points.shape[0],
        controller_points=None,
        reverse_z=reverse_z,
        spring_Y_min=spring_Y_min,
        spring_Y_max=spring_Y_max,
        gt_object_points=torch.from_numpy(data["object_points"]).float().to(device),
        gt_object_visibilities=torch.from_numpy(data["object_visibilities"]).float().to(device),
        gt_object_motions_valid=torch.from_numpy(data["object_motions_valid"]).float().to(device),
        self_collision=False,
        disable_backward=True,
    )


def simulate_fall(
    sim_obj,
    structure_points: torch.Tensor,
    num_original_points: int,
    num_frames: int,
    fall_height: float,
    device: str,
    gravity_acc: float = -9.8,
    floor_penalty: float = 100000.0,
    floor_axis: int = 2,
    num_qp: int = 1000,
    newton_iters: int = 10,
) -> np.ndarray:
    """Simulate free fall from a fixed height and return object point trajectories."""
    wp.init()
    scene = SimplicitsScene(device=device, dtype=torch.float32)
    scene.max_newton_steps = int(newton_iters)

    obj_idx = scene.add_object(sim_obj, num_qp=num_qp)
    gravity_vec = torch.tensor([0.0, 0.0, -gravity_acc], device=device, dtype=torch.float32)
    scene.set_scene_gravity(acc_gravity=gravity_vec)

    base_floor = structure_points[:, floor_axis].min().item() - float(fall_height)
    scene.set_scene_floor(
        floor_height=base_floor,
        floor_axis=floor_axis,
        floor_penalty=floor_penalty,
        flip_floor=True,
    )
    scene.reset_scene()

    all_object_points = []
    for t in range(num_frames):
        if t > 0:
            scene.run_sim_step()
        pred_all = scene.get_object_deformed_pts(obj_idx, points=structure_points)
        all_object_points.append(pred_all[:num_original_points].detach().cpu().numpy())

    return np.array(all_object_points), base_floor, floor_axis


def simulate_fall_springmass(
    points: torch.Tensor,
    num_original_points: int,
    num_frames: int,
    device: str,
    fall_height: float,
    data: Dict[str, Any],
    checkpoint_path: str | None = None,
    case_name: str = "single_push_sloth",
) -> np.ndarray:
    """Simulate free-fall with the Spring-Mass system."""
    cfg.data_type = "synthetic"
    cfg.use_graph = False
    points = points.clone()
    points[:, 2] -= fall_height

    # Read the first-stage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    assert os.path.exists(
        optimal_path
    ), f"Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    simulator = build_springmass_simulator(
        points=points,
        device=device,
        object_radius=cfg.object_radius,
        object_max_neighbours=cfg.object_max_neighbours,
        dt=cfg.dt,
        num_substeps=cfg.num_substeps,
        spring_Y=cfg.init_spring_Y,
        collide_elas=cfg.collide_elas,
        collide_fric=cfg.collide_fric,
        dashpot_damping=cfg.dashpot_damping,
        drag_damping=cfg.drag_damping,
        collide_object_elas=cfg.collide_object_elas,
        collide_object_fric=cfg.collide_object_fric,
        collision_dist=cfg.collision_dist,
        reverse_z=cfg.reverse_z,
        spring_Y_min=cfg.spring_Y_min,
        spring_Y_max=cfg.spring_Y_max,
        data=data,
    )

   
    # Try loading trained per-spring stiffness from checkpoint (best-effort).
    # The checkpoint may have a different spring topology (trained with controller
    # points that we don't use in free-fall), so we skip if counts don't match.
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        spring_Y = checkpoint["spring_Y"]
        if len(spring_Y) == simulator.n_springs:
            simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            logger.info(f"[Fall] Loaded trained spring_Y from {checkpoint_path}")
        else:
            logger.warning(
                f"[Fall] Skipping checkpoint spring_Y: checkpoint has {len(spring_Y)} springs, "
                f"simulator has {simulator.n_springs}. Topology mismatch (likely due to "
                f"controller points used during training). Using optimal_params defaults."
            )

    simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
    # Include initial state as frame 0
    x0 = wp.to_torch(simulator.wp_init_vertices, requires_grad=False)
    all_object_points = [x0.detach().cpu().numpy()[:num_original_points]]

    for _ in range(1, num_frames):
        simulator.step()
        x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
        all_object_points.append(x.detach().cpu().numpy()[:num_original_points])
        simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v)

    floor_height = 0 #points[:, 2].min().item()
    return np.array(all_object_points), floor_height, 2


def render_matplotlib_video(
    object_points: np.ndarray,
    out_video_base: str,
    floor_height: float,
    floor_axis: int,
    object_colors: np.ndarray | None = None,
    max_points: int = 20000,
    viewpoints: List[Tuple[float, float, str]] | None = None,
) -> None:
    """Render matplotlib 3D video(s) with object points and floor plane.

    If *viewpoints* is provided, renders one video per (elev, azim, name) tuple
    with output paths ``{out_video_base}_{name}.mp4``.
    Otherwise renders a single video at defaults.
    """
    if viewpoints is None:
        viewpoints = [(20.0, 120.0, "")]

    num_frames, num_points, _ = object_points.shape
    # 180 degree rotation around X-axis
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
    object_points = object_points @ R.T

    all_pts = object_points.reshape(-1, 3)
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(all_pts).all(axis=1)
    if not valid_mask.any():
        logger.warning("[Render] All points are NaN/Inf, using default bounds")
        mins = np.array([-1.0, -1.0, -1.0])
        maxs = np.array([1.0, 1.0, 1.0])
    else:
        valid_pts = all_pts[valid_mask]
        mins = valid_pts.min(axis=0)
        maxs = valid_pts.max(axis=0)
        pad = 0.05 * (maxs - mins + 1e-6)
        mins -= pad
        maxs += pad

    width, height = (cfg.WH if hasattr(cfg, "WH") else (1024, 768))

    for elev, azim, view_name in viewpoints:
        if view_name:
            video_path = f"{out_video_base}_{view_name}.mp4"
        else:
            video_path = f"{out_video_base}.mp4"

        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cfg.FPS,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        for t in range(num_frames):
            pts = object_points[t]
            
            # Filter out NaN/Inf points in this frame
            valid_mask = np.isfinite(pts).all(axis=1)
            if not valid_mask.any():
                logger.warning(f"[Render] Frame {t}: all points are NaN/Inf, skipping")
                continue
            pts = pts[valid_mask]
            
            if num_points > max_points:
                actual_pts = len(pts)
                sample_size = min(max_points, actual_pts)
                idx = np.random.choice(actual_pts, size=sample_size, replace=False)
                pts = pts[idx]
                if object_colors is not None:
                    frame_colors = object_colors[t][valid_mask]
                    colors = frame_colors[idx]
                else:
                    colors = None
            else:
                colors = object_colors[t][valid_mask] if object_colors is not None else None

            fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)

            # Floor plane
            grid_u = np.linspace(mins[(floor_axis + 1) % 3], maxs[(floor_axis + 1) % 3], 10)
            grid_v = np.linspace(mins[(floor_axis + 2) % 3], maxs[(floor_axis + 2) % 3], 10)
            uu, vv = np.meshgrid(grid_u, grid_v)
            plane = np.zeros((*uu.shape, 3))
            if floor_axis == 0:
                plane[..., 0] = floor_height
                plane[..., 1] = uu
                plane[..., 2] = vv
            elif floor_axis == 1:
                plane[..., 0] = uu
                plane[..., 1] = floor_height
                plane[..., 2] = vv
            else:
                plane[..., 0] = uu
                plane[..., 1] = vv
                plane[..., 2] = floor_height
            ax.plot_surface(
                plane[..., 0],
                plane[..., 1],
                plane[..., 2],
                color=(0.7, 0.7, 0.7),
                alpha=0.4,
                linewidth=0,
            )

            if colors is None:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="red", alpha=0.8)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=colors, alpha=0.8)

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_box_aspect((1, 1, 1))

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.buffer_rgba()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            video_writer.write(frame_bgr)
            plt.close(fig)

        video_writer.release()
        logger.info(f"[Render] Saved PC video: {video_path}")


ORBITAL_VIEWS = [
    (0, "front"),
    (90, "right"),
    (180, "back"),
    (270, "left"),
]

PC_VIEWS = [
    (0, 0, "front"),
    (0, 90, "right"),
    (0, 180, "back"),
    (0, 270, "left"),
]


def create_orbital_cameras(
    object_center: np.ndarray | None,
    gs_source_path: str,
    angles_deg: List[Tuple[float, str]],
    radius: float = 0.6,
    height_offset: float = 0.0,
) -> List[Tuple[Camera, str]]:
    """Create Camera objects orbiting the object center at eye-level.

    Args:
        object_center: (3,) center of object in world coordinates, or None to
            auto-detect from the observation point cloud.
        gs_source_path: path to gaussian data dir (for intrinsics).
        angles_deg: list of (angle_degrees, view_name).
        radius: orbital radius from object center.
        height_offset: vertical offset for camera (negative = move down).

    Returns:
        List of (Camera, view_name) tuples.
    """
    with open(os.path.join(gs_source_path, "camera_meta.pkl"), "rb") as f:
        cam_meta = pickle.load(f)
    K = np.array(cam_meta["intrinsics"][0])
    from PIL import Image
    img_path = os.path.join(gs_source_path, "0.png")
    W, H = Image.open(img_path).size
    FovX = focal2fov(K[0, 0], W)
    FovY = focal2fov(K[1, 1], H)

    if object_center is None:
        # Auto-detect from observation point cloud
        import plyfile
        ply_path = os.path.join(gs_source_path, "observation.ply")
        ply = plyfile.PlyData.read(ply_path)
        xyz = np.stack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], axis=1)
        object_center = xyz.mean(axis=0)

    cameras = []
    for angle_deg, name in angles_deg:
        angle_rad = math.radians(angle_deg)

        # Camera position on a horizontal circle around object center
        cam_pos = np.array([
            object_center[0] + radius * math.cos(angle_rad),
            object_center[1] + radius * math.sin(angle_rad),
            object_center[2] + height_offset,  # apply vertical offset
        ], dtype=np.float64)

        # Look-at: camera Z-axis points from camera toward object (GS convention: +Z = forward)
        forward = object_center - cam_pos
        forward = forward / np.linalg.norm(forward)

        # World up: in PhysTwin, objects are at negative Z (above floor at Z=0)
        # Existing cameras have up vectors with positive Z component
        world_up = np.array([0.0, 0.0, 1.0])

        # Right vector (X = Y x Z in right-handed: right = up x forward)
        right = np.cross(world_up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        # Recompute up (Y = Z x X: up = forward x right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # C2W matrix: columns are right, up, forward (GS convention: +Z = look direction)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos

        # W2C = inv(C2W)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # stored transposed for GS
        T = w2c[:3, 3]

        cam = Camera(
            resolution=(W, H),
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name=f"orbital_{name}",
            uid=0,
            K=K,
        )
        cameras.append((cam, name))

    return cameras


def render_gaussian_video(
    gs_model_path: str,
    gs_source_path: str,
    out_video_base: str,
    object_points: np.ndarray,
    iteration: int = -1,
    white_background: bool = True,
    views: List[Tuple[Camera, str]] | None = None,
    view_idx: int = 0,
) -> None:
    """Render RGB video(s) using Gaussian Splatting driven by object point trajectories.

    If *views* is provided, renders one video per (camera, name) pair with output
    paths ``{out_video_base}_{name}.mp4``.  Otherwise falls back to loading the
    scene's test cameras and using *view_idx*.
    """
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    gs_args = parser.parse_args([])
    gs_args.source_path = os.path.abspath(gs_source_path)
    gs_args.model_path = gs_model_path
    gs_args.white_background = white_background
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

    # Determine camera list
    if views is not None:
        camera_list = views
    else:
        test_cams = scene_obj.getTestCameras()
        idx = int(view_idx)
        if idx < 0 or idx >= len(test_cams):
            raise ValueError(f"view_idx {idx} out of range (0..{len(test_cams)-1})")
        camera_list = [(test_cams[idx], "")]

    # Compute all frame Gaussian states once
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

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render each viewpoint
    for cam, view_name in camera_list:
        if view_name:
            video_path = f"{out_video_base}_{view_name}.mp4"
        else:
            video_path = f"{out_video_base}.mp4"

        h, w = cam.image_height, cam.image_width
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.FPS, (w, h)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        for i in range(n_steps):
            gaussians._xyz = xyz[i].to("cuda")
            gaussians._features_dc = rgb[i].unsqueeze(1).to("cuda")
            gaussians._rotation = quat[i].to("cuda")
            gaussians._opacity = gaussians.inverse_opacity_activation(opa[i]).to("cuda")

            results = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
            rendering = results["render"][:3].detach().cpu().permute(1, 2, 0).numpy()
            frame = (np.clip(rendering, 0.0, 1.0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        logger.info(f"[Render] Saved RGB video: {video_path}")


def main():
    parser = ArgumentParser(description="Generate synthetic fall data using PhysTwin digital twin")
    parser.add_argument("--data_path", type=str, required=True, help="Reference data pickle")
    parser.add_argument("--sim_obj_path", type=str, default=None, help="Path to best_sim_obj.pkl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save synthetic data pickle")
    parser.add_argument("--simulator", type=str, default="simplicits",
                        choices=["simplicits", "springmass"],
                        help="Physics backend to use for fall simulation")
    parser.add_argument("--springmass_checkpoint", type=str, default=None,
                        help="Optional checkpoint for Spring-Mass parameters (best_*.pth)")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--fall_height", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_qp", type=int, default=1000)
    parser.add_argument("--floor_penalty", type=float, default=100000.0)
    parser.add_argument("--newton_iters", type=int, default=10)
    parser.add_argument("--gravity_acc", type=float, default=-9.8)
    parser.add_argument("--render_rgb", action="store_true", default=False)
    parser.add_argument("--gs_model_path", type=str, default=None)
    parser.add_argument("--gs_source_path", type=str, default=None)
    parser.add_argument("--gs_iteration", type=int, default=-1)
    parser.add_argument("--gs_view_idx", type=int, default=0)
    parser.add_argument("--gs_white_background", action="store_true", default=True)
    parser.add_argument("--camera_height_offset", type=float, default=-0.15,
                        help="Vertical offset for camera position (negative = move down)")
    parser.add_argument("--render_pointcloud", action="store_true", default=False,
                        help="Render matplotlib3d point cloud video with floor plane")
    parser.add_argument("--pc_vis_cam_idx", type=int, default=0)
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--vid2sim_transforms", type=str, default=None,
                        help="Path to Vid2Sim transforms_simulation.json for camera layout")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger.set_log_file(path=os.path.dirname(args.output_path), name="synthetic_fall_log")

    cfg.device = args.device

    ref = load_reference_data(args.data_path, args.device)
    structure_points = ref["structure_points"]
    num_original_points = ref["num_original_points"]

    if args.simulator == "simplicits":
        logger.info(f"[Fall] Loading SimplicitsObject from {args.sim_obj_path}")
        with open(args.sim_obj_path, "rb") as f:
            sim_obj = pickle.load(f)

        object_points_array, floor_height, floor_axis = simulate_fall(
            sim_obj=sim_obj,
            structure_points=structure_points,
            num_original_points=num_original_points,
            num_frames=args.num_frames,
            fall_height=args.fall_height,
            device=args.device,
            gravity_acc=args.gravity_acc,
            floor_penalty=args.floor_penalty,
            floor_axis=2,
            num_qp=args.num_qp,
            newton_iters=args.newton_iters,
            
        )
    else:
        object_points_array, floor_height, floor_axis = simulate_fall_springmass(
            points=structure_points,
            num_original_points=num_original_points,
            num_frames=args.num_frames,
            device=args.device,
            fall_height=args.fall_height,
            data=ref['raw'],
            checkpoint_path=args.springmass_checkpoint,
            case_name=args.case_name or "single_push_sloth",
        )

    object_visibilities = np.ones(
        (args.num_frames, num_original_points), dtype=bool
    )
    object_motions_valid = np.ones_like(object_visibilities, dtype=bool)

    ref_raw = ref["raw"]
    object_colors = ref_raw["object_colors"]

    object_colors = np.tile(object_colors[0:1], (args.num_frames, 1, 1))

    controller_points = ref_raw.get("controller_points", None)
    if controller_points is None:
        controller_points = np.zeros((args.num_frames, 0, 3), dtype=np.float32)

    synthetic_data = {
        "object_points": object_points_array,
        "object_colors": object_colors,
        "object_visibilities": object_visibilities,
        "object_motions_valid": object_motions_valid,
        "controller_points": None,
        "surface_points": ref_raw["surface_points"],
        "interior_points": ref_raw["interior_points"],
        "fall_height": float(args.fall_height),
        "floor_height": float(floor_height),
        "floor_axis": int(floor_axis),
        "num_original_points": int(num_original_points),
        "frame_len": int(args.num_frames),
    }

    with open(args.output_path, "wb") as f:
        pickle.dump(synthetic_data, f)
    logger.info(f"[Fall] Saved synthetic data to {args.output_path}")

    if args.render_pointcloud:
        if args.base_path is not None and args.case_name is not None:
            load_camera_meta(args.base_path, args.case_name)
        pc_base = os.path.join(os.path.dirname(args.output_path), "synthetic_fall_pc")
        logger.info(f"[Render] Rendering matplotlib3d videos (4 views)")
        render_matplotlib_video(
            object_points=object_points_array,
            out_video_base=pc_base,
            floor_height=floor_height,
            floor_axis=floor_axis,
            object_colors=object_colors,
            viewpoints=PC_VIEWS,
        )
        logger.info("[Render] Matplotlib3d videos complete.")

    if args.render_rgb:
        # Free GPU memory from simulation before loading Gaussian model.
        # Warp and PyTorch both hold GPU allocations; aggressively release everything.
        import gc
        del ref, ref_raw, synthetic_data, structure_points
        if 'object_colors' in dir():
            del object_colors
        gc.collect()
        torch.cuda.empty_cache()
        # Warp caches GPU memory outside PyTorch — synchronize to flush
        wp.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if not args.gs_model_path or not args.gs_source_path:
            raise ValueError("gs_model_path and gs_source_path are required when render_rgb is set.")

        if args.vid2sim_transforms:
            # Use Vid2Sim 12-camera layout (same as generate_vid2sim_data.py)
            from generate_vid2sim_data import (
                load_vid2sim_camera_transforms,
                compute_vid2sim_reference_center_and_scale,
                create_vid2sim_cameras,
            )
            vid2sim_c2ws = load_vid2sim_camera_transforms(args.vid2sim_transforms)
            vid2sim_center, vid2sim_avg_dist = compute_vid2sim_reference_center_and_scale(vid2sim_c2ws)

            all_pts = object_points_array.reshape(-1, 3)
            obj_min, obj_max = all_pts.min(axis=0), all_pts.max(axis=0)
            obj_center = (obj_min + obj_max) / 2.0
            obj_extent = np.linalg.norm(obj_max - obj_min)

            vid2sim_cams_raw = create_vid2sim_cameras(
                object_center=obj_center,
                object_extent=obj_extent,
                gs_source_path=os.path.abspath(args.gs_source_path),
                vid2sim_c2ws=vid2sim_c2ws,
                vid2sim_center=vid2sim_center,
                vid2sim_avg_dist=vid2sim_avg_dist,
            )
            # convert (Camera, view_idx) to (Camera, str_name) for render_gaussian_video
            render_views = [(cam, f"view{idx}") for cam, idx in vid2sim_cams_raw]
            logger.info(f"[Render] Using Vid2Sim camera layout ({len(render_views)} views)")
        else:
            # Fallback: 4 orbital cameras
            render_views = create_orbital_cameras(
                object_center=None,  # auto-detect from GS model
                gs_source_path=os.path.abspath(args.gs_source_path),
                angles_deg=ORBITAL_VIEWS,
                height_offset=args.camera_height_offset,
            )
            logger.info(f"[Render] Using orbital camera layout ({len(render_views)} views)")

        rgb_base = os.path.join(os.path.dirname(args.output_path), "synthetic_fall_rgb")
        render_gaussian_video(
            gs_model_path=args.gs_model_path,
            gs_source_path=args.gs_source_path,
            out_video_base=rgb_base,
            object_points=object_points_array,
            iteration=args.gs_iteration,
            white_background=args.gs_white_background,
            views=render_views,
        )
        logger.info("[Render] RGB videos complete.")


if __name__ == "__main__":
    main()
