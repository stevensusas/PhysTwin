#!/usr/bin/env python3
import os
import json
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warp as wp

import kaolin.physics.utils.warp_utilities as warp_utilities
from kaolin.physics.simplicits.easy_api import SimplicitsObject
from kaolin.physics.simplicits.precomputed import sparse_lbs_matrix

from qqtt.utils import cfg, logger
from qqtt.data import RealData

from solver_ours_new import BatchedSimplicitsSolver

# Optional visualization helper
from script_generate_fall_synthetic import render_matplotlib_video

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
# Import helpers from inference script (boundary mapping & scene setup)
import importlib.util

inference_script_path = os.path.join(os.path.dirname(__file__), "script_inference_simplicit_easy_api.py")
spec = importlib.util.spec_from_file_location("inference_module", inference_script_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)

extract_boundary_constraints = inference_module.extract_boundary_constraints
setup_scene_with_forces = inference_module.setup_scene_with_forces


class SimpleDataset:
    """Lightweight dataset wrapper for synthetic pickle with RealData keys."""

    def __init__(self, data_path: str, device: str):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.object_points = torch.tensor(data["object_points"], dtype=torch.float32, device=device)
        self.object_colors = torch.tensor(data["object_colors"], dtype=torch.float32, device=device)
        self.object_visibilities = torch.tensor(data["object_visibilities"], dtype=torch.bool, device=device)
        self.object_motions_valid = torch.tensor(data["object_motions_valid"], dtype=torch.bool, device=device)
        self.controller_points = (
            torch.tensor(data["controller_points"], dtype=torch.float32, device=device)
            if data.get("controller_points") is not None
            else None
        )

        surface_points = torch.tensor(data["surface_points"], dtype=torch.float32, device=device)
        interior_points = torch.tensor(data["interior_points"], dtype=torch.float32, device=device)
        self.structure_points = torch.cat([self.object_points[0], surface_points, interior_points], dim=0)
        self.num_original_points = self.object_points.shape[1]
        self.frame_len = self.object_points.shape[0]


class MaterialMLP(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, initial_E=1e5, initial_nu=0.3, bounds=None):
        super().__init__()
        self.bounds = bounds
        self.log_E_min = np.log(bounds["E"][0])
        self.log_E_max = np.log(bounds["E"][1])

        layers = []
        input_dim = 3
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)

        target_log = np.log(initial_E)
        norm_val_E = (target_log - self.log_E_min) / (self.log_E_max - self.log_E_min)
        norm_val_E = max(min(norm_val_E, 0.99), 0.01)
        nu_norm = (initial_nu - bounds["nu"][0]) / (bounds["nu"][1] - bounds["nu"][0])
        nu_norm = max(min(nu_norm, 0.99), 0.01)

        with torch.no_grad():
            self.head.bias[0] = np.log(norm_val_E / (1 - norm_val_E))
            self.head.bias[1] = np.log(nu_norm / (1 - nu_norm))

    def forward(self, x):
        feats = self.net(x)
        raw = self.head(feats)
        log_E = self.log_E_min + (self.log_E_max - self.log_E_min) * torch.sigmoid(raw[:, 0])
        E = torch.exp(log_E)
        nu = self.bounds["nu"][0] + (self.bounds["nu"][1] - self.bounds["nu"][0]) * torch.sigmoid(raw[:, 1])
        return E, nu


class GripMLP(nn.Module):
    """Grip potential on constraint points; outputs [0,1]."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_lame_params_field(E, nu):
    nu = torch.clamp(nu, 0.01, 0.49)
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def build_simplicits_object_from_params(structure_points, E, nu, rho, device, dtype, num_handles, training_num_steps):
    n_points = structure_points.shape[0]
    yms = E.expand(n_points) if E.dim() == 0 else E
    prs = nu.expand(n_points) if nu.dim() == 0 else nu
    rhos = rho.expand(n_points) if rho.dim() == 0 else rho
    bbox_min = structure_points.min(dim=0)[0]
    bbox_max = structure_points.max(dim=0)[0]
    bbox_size = (bbox_max - bbox_min).abs()
    approx_volume = float((bbox_size[0] * bbox_size[1] * bbox_size[2]).detach().cpu().item())
    return SimplicitsObject.create_trained(
        structure_points,
        yms,
        prs,
        rhos,
        torch.tensor([approx_volume], dtype=dtype, device=device),
        num_handles=num_handles,
        training_num_steps=training_num_steps,
        training_le_coeff=0.1,
    )


def compute_chamfer_loss(pred_pts, gt_pts, vis_mask, chamfer_weight=1.0):
    if pred_pts.dim() == 2:
        pred_pts = pred_pts.unsqueeze(0)
        gt_pts = gt_pts.unsqueeze(0)
        vis_mask = vis_mask.unsqueeze(0)

    total_loss = torch.tensor(0.0, device=pred_pts.device, dtype=pred_pts.dtype)
    total_valid = 0

    for b in range(pred_pts.shape[0]):
        valid_mask = vis_mask[b]
        if valid_mask.sum() == 0:
            continue
        visible_pred = pred_pts[b][valid_mask]
        visible_gt = gt_pts[b][valid_mask]
        pred_expanded = visible_pred.unsqueeze(1)
        gt_expanded = visible_gt.unsqueeze(0)
        distances_sq = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)
        min_distances_sq, _ = torch.min(distances_sq, dim=1)
        chamfer_loss = chamfer_weight * torch.mean(min_distances_sq)
        total_loss += chamfer_loss
        total_valid += 1

    return total_loss / max(total_valid, 1)


def compute_tracking_loss(pred_pts, gt_pts, vis_mask, track_weight=1.0):
    if pred_pts.dim() == 2:
        pred_pts = pred_pts.unsqueeze(0)
        gt_pts = gt_pts.unsqueeze(0)
        vis_mask = vis_mask.unsqueeze(0)

    total_loss = torch.tensor(0.0, device=pred_pts.device, dtype=pred_pts.dtype)
    total_valid = 0

    for b in range(pred_pts.shape[0]):
        valid_mask = vis_mask[b]
        if valid_mask.sum() == 0:
            continue
        valid_pred = pred_pts[b][valid_mask]
        valid_gt = gt_pts[b][valid_mask]
        diff = torch.abs(valid_pred - valid_gt)
        track_loss_val = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
        track_loss_sum = track_loss_val.sum()
        average_factor = float(valid_mask.sum().item()) * 3.0
        total_loss += track_weight * track_loss_sum / average_factor
        total_valid += 1

    return total_loss / max(total_valid, 1)


@torch.no_grad()
def validate_sequence(
    solver,
    sim_obj,
    object_points_rest,
    gt_positions,
    gt_visibilities,
    fixed_sim_points,
    mlp,
    rho_raw,
    bounds,
    per_frame_hand_targets,
    constraint_indices_torch,
    floor_penalty,
    boundary_penalty,
    has_constraints,
    chamfer_weight,
    track_weight,
):
    E_field, nu_field = mlp(fixed_sim_points)
    rho_val = bounds["rho"][0] + (bounds["rho"][1] - bounds["rho"][0]) * torch.sigmoid(rho_raw)
    mu_field, lam_field = compute_lame_params_field(E_field, nu_field)
    solver.set_density(rho_val)

    z = torch.zeros((1, solver.num_dof), device=object_points_rest.device)
    v = torch.zeros_like(z)
    dt_sub = solver.frame_dt

    total_loss = 0.0
    for t in range(gt_positions.shape[0]):
        penalty_list = []
        if has_constraints and t < len(per_frame_hand_targets):
            penalty_list = [(
                constraint_indices_torch,
                per_frame_hand_targets[t].unsqueeze(0),
                boundary_penalty,
            )]
        if t > 0:
            z_prev = z.clone()
            z = solver.newton_step_batch(
                z, v, mu_field, lam_field, penalty_list, dt_sub,
                floor_penalty=floor_penalty, num_iters=10
            )
            v = (z - z_prev) / dt_sub

        pred_pts = solver.get_full_points_batch(object_points_rest, z, sim_obj.skinning_weight_function)
        gt = gt_positions[t]
        vis = gt_visibilities[t]
        total_loss += (
            compute_chamfer_loss(pred_pts, gt, vis, chamfer_weight=chamfer_weight)
            + compute_tracking_loss(pred_pts, gt, vis, track_weight=track_weight)
        )

    return float(total_loss / max(gt_positions.shape[0], 1))


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="outputs/joint_optim_clean")
    parser.add_argument("--base_path", type=str, default="/media/extra_disk/data/different_types")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=int, default=84)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--initial_E", type=float, default=1e5)
    parser.add_argument("--initial_nu", type=float, default=0.4)
    parser.add_argument("--initial_rho", type=float, default=500.0)
    parser.add_argument("--floor_penalty", type=float, default=1e5)
    parser.add_argument("--boundary_penalty", type=float, default=5000.0)
    parser.add_argument("--num_handles", type=int, default=10)
    parser.add_argument("--num_qp", type=int, default=500)
    parser.add_argument("--training_num_steps", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="physTwin_joint_optim_clean")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--vis_interval", type=int, default=50)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--supervised_lbs", action="store_true", default=False)
    parser.add_argument("--lbs_optim_interval", type=int, default=5)
    parser.add_argument("--lbs_training_steps", type=int, default=1000)
    parser.add_argument("--chamfer_weight", type=float, default=1.0)
    parser.add_argument("--track_weight", type=float, default=1.0)
    parser.add_argument("--learn_grip", action="store_true", default=False)
    parser.add_argument("--grip_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--validation_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--no_boundary", action="store_true", help="Disable boundary constraints even if controllers exist.")
    parser.add_argument("--data_type", type=str, default="real", choices=["real", "synthetic"])
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"{args.base_dir}/{args.case_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.set_log_file(path=args.output_dir, name="joint_optim_clean_log")

    cfg.data_path = args.data_path
    cfg.base_dir = args.base_dir
    cfg.device = args.device

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            dir=args.output_dir,
        )

    # Load dataset
    if args.data_type == "real":
        dataset = RealData(visualize=False, save_gt=False)
    else:
        dataset = SimpleDataset(args.data_path, args.device)

    structure_points = dataset.structure_points.to(args.device)
    gt_positions = dataset.object_points[: args.num_frames].to(args.device)
    gt_visibilities = dataset.object_visibilities[: args.num_frames].to(args.device)
    object_points_rest = dataset.object_points[0].to(args.device)

    constraint_point_indices, per_frame_hand_targets = extract_boundary_constraints(dataset)
    if args.no_boundary:
        constraint_point_indices = torch.zeros((0,), dtype=torch.int32, device=args.device)
        per_frame_hand_targets = torch.zeros((gt_positions.shape[0], 0, 3), device=args.device)

    has_constraints = (
        constraint_point_indices is not None
        and len(constraint_point_indices) > 0
        and per_frame_hand_targets is not None
        and per_frame_hand_targets.numel() > 0
    )

    # Build initial object
    sim_obj = build_simplicits_object_from_params(
        structure_points,
        torch.tensor(args.initial_E, device=args.device),
        torch.tensor(args.initial_nu, device=args.device),
        torch.tensor(args.initial_rho, device=args.device),
        device=args.device,
        dtype=torch.float32,
        num_handles=args.num_handles,
        training_num_steps=args.training_num_steps,
    )

    # Scene setup
    initial_targets = per_frame_hand_targets[0] if has_constraints else None
    scene, _, _, _, floor_height, floor_axis, cubature_constraint_indices = setup_scene_with_forces(
        sim_obj,
        constraint_point_indices,
        initial_targets,
        structure_points,
        device=args.device,
        gravity_acc=-9.8,
        boundary_penalty=args.boundary_penalty,
        floor_penalty=args.floor_penalty,
        sample_indices=None,
    )

    if isinstance(cubature_constraint_indices, wp.array):
        constraint_indices_torch = wp.to_torch(cubature_constraint_indices).to(dtype=torch.long, device=args.device)
    else:
        constraint_indices_torch = cubature_constraint_indices.to(dtype=torch.long, device=args.device)

    fixed_sim_points = wp.to_torch(scene.sim_pts).clone().to(args.device)

    with torch.no_grad():
        full_weights = sim_obj.skinning_weight_function(structure_points)
        full_B_wp = sparse_lbs_matrix(wp.from_torch(full_weights), wp.from_torch(structure_points, dtype=wp.vec3))
        B_full_dense = warp_utilities._bsr_to_torch(full_B_wp).to_dense().to(args.device)
        if B_full_dense.shape[0] > 3 * gt_positions.shape[1]:
            B_full_dense = B_full_dense[: 3 * gt_positions.shape[1], :]

    solver = BatchedSimplicitsSolver(
        scene,
        B_full_dense,
        args.initial_rho,
        args.device,
        floor_height=0.0,
        gravity_acc=-9.8,
        floor_axis=floor_axis,
    )

    bounds = {"E": (1e3, 5e5), "nu": (0.35, 0.49), "rho": (1e2, 1e4)}
    mlp = MaterialMLP(initial_E=args.initial_E, initial_nu=args.initial_nu, bounds=bounds).to(args.device)
    grip_mlp = GripMLP(hidden_dim=64).to(args.device) if args.learn_grip else None
    rho_raw = nn.Parameter(torch.logit(torch.tensor(0.5, device=args.device)))
    params = list(mlp.parameters()) + [rho_raw]
    if grip_mlp is not None:
        params += list(grip_mlp.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    history = []
    best_loss = float("inf")
    best_val = float("inf")

    num_substeps = 1
    dt_sub = solver.frame_dt / num_substeps

    pbar = tqdm(range(args.iters))
    for iter_idx in pbar:
        optimizer.zero_grad()

        # Optional supervised LBS retraining
        if args.supervised_lbs and iter_idx > 0 and iter_idx % args.lbs_optim_interval == 0:
            from script_joint_optimize_simplicit_easy_api_torch import retrain_lbs_network_supervised
            with torch.no_grad():
                E_cub, nu_cub = mlp(fixed_sim_points)
                mu_cub, lam_cub = compute_lame_params_field(E_cub, nu_cub)
                rho_val = bounds["rho"][0] + (bounds["rho"][1] - bounds["rho"][0]) * torch.sigmoid(rho_raw)

            z_init_for_lbs = torch.zeros((solver.num_dof,), device=args.device, dtype=torch.float32)
            sim_obj = retrain_lbs_network_supervised(
                sim_obj,
                structure_points,
                E_cub.unsqueeze(-1),
                nu_cub.unsqueeze(-1),
                torch.full_like(E_cub.unsqueeze(-1), rho_val),
                solver,
                z_init_for_lbs,
                mu_cub,
                lam_cub,
                rho_val,
                gt_positions,
                gt_visibilities,
                per_frame_hand_targets,
                constraint_indices_torch,
                args.boundary_penalty,
                dt_sub,
                args.floor_penalty,
                device=args.device,
                num_handles=args.num_handles,
                training_num_steps=args.lbs_training_steps,
                num_substeps=num_substeps,
                chamfer_weight=args.chamfer_weight,
                track_weight=args.track_weight,
                acc_weight=1.0,
                normalize_for_training=True,
                friction_coeff=0.0,
                constraint_offset=None,
                object_points_rest=object_points_rest,
            )
        # Random window start
        window_size = min(args.window_size, gt_positions.shape[0])
        max_start = max(gt_positions.shape[0] - window_size, 0)
        start_frames = torch.randint(0, max_start + 1, (args.batch_size,), device=args.device)

        # Initialize from rest and fast-forward to start_frames (no grad)
        with torch.no_grad():
            E_init, nu_init = mlp(fixed_sim_points)
            rho_val = bounds["rho"][0] + (bounds["rho"][1] - bounds["rho"][0]) * torch.sigmoid(rho_raw)
            mu_field, lam_field = compute_lame_params_field(E_init, nu_init)
            solver.set_density(rho_val)

            z_batch = torch.zeros((args.batch_size, solver.num_dof), device=args.device)
            v_prev_batch = torch.zeros_like(z_batch)
            max_start_frame = int(start_frames.max().item()) if start_frames.numel() > 0 else 0

            for s in range(max_start_frame):
                penalty_list = []
                if has_constraints:
                    step_targets = per_frame_hand_targets[s]
                    k_pen = args.boundary_penalty
                    if grip_mlp is not None:
                        grip_vals = grip_mlp(fixed_sim_points[constraint_indices_torch])
                        k_pen = grip_vals
                    penalty_list = [(
                        constraint_indices_torch,
                        step_targets.unsqueeze(0).expand(args.batch_size, -1, -1),
                        k_pen,
                    )]
                z_next = solver.newton_step_batch(
                    z_batch,
                    v_prev_batch,
                    mu_field,
                    lam_field,
                    penalty_list,
                    dt_sub,
                    floor_penalty=args.floor_penalty,
                    num_iters=10,
                )
                v_next = (z_next - z_batch) / dt_sub
                active_mask = (s + 1) < start_frames
                z_batch = torch.where(active_mask.unsqueeze(1), z_next, z_batch)
                v_prev_batch = torch.where(active_mask.unsqueeze(1), v_next, v_prev_batch)

        z_batch.requires_grad_(True)

        loss_total = 0.0
        chamfer_total = 0.0
        track_total = 0.0
        for i in range(window_size):
            E_field, nu_field = mlp(fixed_sim_points)
            rho_val = bounds["rho"][0] + (bounds["rho"][1] - bounds["rho"][0]) * torch.sigmoid(rho_raw)
            mu_field, lam_field = compute_lame_params_field(E_field, nu_field)
            solver.set_density(rho_val)

            current_times = start_frames + i
            penalty_list = []
            if has_constraints:
                batch_targets = per_frame_hand_targets[current_times]
                k_pen = args.boundary_penalty
                if grip_mlp is not None:
                    grip_vals = grip_mlp(fixed_sim_points[constraint_indices_torch])
                    k_pen = grip_vals
                penalty_list = [(
                    constraint_indices_torch,
                    batch_targets,
                    k_pen,
                )]

            z_prev = z_batch.clone()
            for _ in range(num_substeps):
                z_batch = solver.newton_step_batch(
                    z_batch,
                    v_prev_batch,
                    mu_field,
                    lam_field,
                    penalty_list,
                    dt_sub,
                    floor_penalty=args.floor_penalty,
                    num_iters=10,
                )
                v_prev_batch = (z_batch - z_prev) / dt_sub
                z_prev = z_batch

            pred_pts = solver.get_full_points_batch(object_points_rest, z_batch, sim_obj.skinning_weight_function)
            #print(gt_positions.shape, gt_visibilities.shape, "gt_positions and gt_visibilities")
            gt_stack = gt_positions[current_times][0]
            vis_stack = gt_visibilities[current_times][0]
            #print(gt_stack.shape, vis_stack.shape, "gt_stack and vis_stack")
            chamfer_loss = compute_chamfer_loss(
                pred_pts, gt_stack, vis_stack, chamfer_weight=args.chamfer_weight
            )
            track_loss = compute_tracking_loss(
                pred_pts, gt_stack, vis_stack, track_weight=args.track_weight
            )
            loss_frame = chamfer_loss + track_loss
            if grip_mlp is not None and args.grip_sparsity_weight > 0.0 and has_constraints:
                grip_vals = grip_mlp(fixed_sim_points[constraint_indices_torch])
                loss_frame = loss_frame + args.grip_sparsity_weight * grip_vals.mean()
            loss_total = loss_total + loss_frame
            chamfer_total = chamfer_total + chamfer_loss
            track_total = track_total + track_loss

        loss_avg = loss_total / window_size
        chamfer_avg = chamfer_total / window_size
        track_avg = track_total / window_size
        loss_avg.backward()
        optimizer.step()

        pbar.set_description(f"TrainL={loss_avg.item():.4f}")
        history.append({
            "iter": iter_idx,
            "loss": loss_avg.item(),
            "chamfer_loss": chamfer_avg.item(),
            "track_loss": track_avg.item(),
            "rho": float(rho_val.item()),
            "E_mean": float(E_field.mean().item()),
            "nu_mean": float(nu_field.mean().item()),
        })

        if iter_idx % args.validation_interval == 0:
            val_loss = validate_sequence(
                solver,
                sim_obj,
                object_points_rest,
                gt_positions,
                gt_visibilities,
                fixed_sim_points,
                mlp,
                rho_raw,
                bounds,
                per_frame_hand_targets,
                constraint_indices_torch,
                args.floor_penalty,
                args.boundary_penalty,
                has_constraints,
                args.chamfer_weight,
                args.track_weight,
            )
            best_val = min(best_val, val_loss)
            if val_loss <= best_val:
                best_ckpt = {
                    "iter": int(iter_idx),
                    "mlp_state": mlp.state_dict(),
                    "rho": float(rho_val.item()),
                    "E_mean": float(E_field.mean().item()),
                    "nu_mean": float(nu_field.mean().item()),
                    "chamfer_weight": args.chamfer_weight,
                    "track_weight": args.track_weight,
                }
                if grip_mlp is not None:
                    best_ckpt["grip_mlp"] = grip_mlp.state_dict()
                torch.save(best_ckpt, os.path.join(args.output_dir, "best_material_mlp.pth"))

        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                "train_loss": loss_avg.item(),
                "train_chamfer_loss": chamfer_avg.item(),
                "train_track_loss": track_avg.item(),
                "rho": float(rho_val.item()),
                "E_mean": float(E_field.mean().item()),
                "nu_mean": float(nu_field.mean().item()),
                "best_val_loss": best_val,
            }
            if iter_idx % args.validation_interval == 0:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict, step=iter_idx)

        if args.vis_interval > 0 and (iter_idx + 1) % args.vis_interval == 0:
            vis_dir = args.vis_dir or os.path.join(args.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"train_iter_{iter_idx+1:05d}.mp4")
            # quick visualization using current predicted points (single rollout)
            with torch.no_grad():
                E_field, nu_field = mlp(fixed_sim_points)
                rho_val = bounds["rho"][0] + (bounds["rho"][1] - bounds["rho"][0]) * torch.sigmoid(rho_raw)
                mu_field, lam_field = compute_lame_params_field(E_field, nu_field)
                solver.set_density(rho_val)
                z = torch.zeros((1, solver.num_dof), device=args.device)
                v = torch.zeros_like(z)
                frames = []
                for t in range(gt_positions.shape[0]):
                    penalty_list = []
                    if has_constraints and t < len(per_frame_hand_targets):
                        penalty_list = [(
                            constraint_indices_torch,
                            per_frame_hand_targets[t].unsqueeze(0),
                            args.boundary_penalty,
                        )]
                    if t > 0:
                        z_prev = z.clone()
                        z = solver.newton_step_batch(
                            z, v, mu_field, lam_field, penalty_list, dt_sub,
                            floor_penalty=args.floor_penalty, num_iters=10
                        )
                        v = (z - z_prev) / dt_sub
                    pred_pts = solver.get_full_points_batch(object_points_rest, z, sim_obj.skinning_weight_function)
                    frames.append(pred_pts.squeeze(0).detach().cpu().numpy())
                render_matplotlib_video(
                    object_points=np.stack(frames, axis=0),
                    out_video_path=vis_path,
                    floor_height=0.0,
                    floor_axis=floor_axis,
                    object_colors=None,
                )

        if args.output_dir and args.checkpoint_interval > 0 and (iter_idx + 1) % args.checkpoint_interval == 0:
            ckpt = {
                "iter": int(iter_idx),
                "mlp_state": mlp.state_dict(),
                "rho": float(rho_val.item()),
                "E_mean": float(E_field.mean().item()),
                "nu_mean": float(nu_field.mean().item()),
                "chamfer_weight": args.chamfer_weight,
                "track_weight": args.track_weight,
            }
            if grip_mlp is not None:
                ckpt["grip_mlp"] = grip_mlp.state_dict()
            torch.save(ckpt, os.path.join(args.output_dir, f"material_mlp_iter_{iter_idx+1:05d}.pth"))

    with open(os.path.join(args.output_dir, "optimization_results.json"), "w") as f:
        json.dump({"optimization_history": history, "best_loss": best_loss, "best_val_loss": best_val}, f, indent=2)

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
