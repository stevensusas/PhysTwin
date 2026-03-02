
"""
Torch-based Batched Newton Solver for Kaolin Simplicits (compatible with compare_solvers.py).

This module provides:
    - BatchedSimplicitsSolver(scene, B_full_dense, rho_init, device, floor_height, gravity_acc, floor_axis)
    - BatchedSimplicitsSolver.newton_step_batch(...)

Design goals:
1) Match Kaolin Easy API (legacy torch version) as closely as possible:
   - world positions: x = B @ z + x0 (x0 = sim_pts)
   - deformation gradients: F = dFdz @ z + bigI
   - backward-Euler / variational timestep objective: E = inertial + dt^2 * potential
   - gravity/floor/boundary energies consistent with Kaolin kernels:
       gravity:    rho * sum(vol * dot(g, x))
       floor:      k * sum(vol * pen^2)      (no 0.5)
       boundary:   k * sum(||x_i - tgt||^2) (no vol)
2) Be differentiable in torch (for system ID), avoiding Warp autograd.
3) Be API-compatible with your running script (compare_solvers.py):
   - has .B_sim attribute
   - exposes newton_step_batch signature including the extra debug kwargs:
       more_partial_newton_E/G/H (ignored, accepted for compatibility)

Notes:
- This solver uses analytic gradients/Hessians for:
    - inertia (dense M in reduced space)
    - Neo-Hookean elastic (via kaolin legacy torch formulas in neohookean_elastic_material_torch.py)
    - gravity / floor / boundary (closed-form in x-space + mapping through B)
  So we do NOT use torch.autograd.functional.hessian (which is slow/unstable for large dof).
- Projection (P/Pt) and bounds (collision) are not required by your compare script; we accept P/Pt
  args for compatibility but default to None. If you want full Kaolin parity (projected solve + bounds),
  we can extend this, but it wasn’t used in your provided driver.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import warp as wp
import kaolin.physics.utils.warp_utilities as warp_utilities

from neohookean_elastic_material_torch import (
    unbatched_neohookean_energy,
    unbatched_neohookean_gradient,
    unbatched_neohookean_hessian,
)

Tensor = torch.Tensor

from kaolin.physics.simplicits.skinning import weight_function_lbs

@dataclass
class NewtonOptions:
    max_iters: int = 10
    conv_tol: float = 1e-4
    line_search_max_steps: int = 10
    line_search_alpha: float = 1e-3
    line_search_beta: float = 0.6
    hessian_regularizer: float = 1e-4
    max_reg_tries: int = 8
    reg_growth: float = 10.0


def _bsr_to_dense(bsr, device: str) -> Tensor:
    # Kaolin stores many sim matrices as Warp BSR. Convert to dense torch.
    return warp_utilities._bsr_to_torch(bsr).to_dense().to(device)


def _safe_solve(H: Tensor, g: Tensor, opts: NewtonOptions) -> Tensor:
    """
    Solve (H + reg I) dx = -g robustly with increasing diagonal damping if needed.
    """
    n = H.shape[0]
    I = torch.eye(n, device=H.device, dtype=H.dtype)

    reg = float(opts.hessian_regularizer)
    for _ in range(opts.max_reg_tries):
        H_reg = H + reg * I
        # Try Cholesky (SPD); if fails fallback to solve.
        try:
            L = torch.linalg.cholesky(H_reg)
            dx = torch.cholesky_solve((-g).unsqueeze(1), L).squeeze(1)
            if torch.isfinite(dx).all():
                return dx
        except RuntimeError:
            pass

        try:
            dx = torch.linalg.solve(H_reg, -g)
            if torch.isfinite(dx).all():
                return dx
        except RuntimeError:
            pass

        reg *= float(opts.reg_growth)

    # Final fallback: least squares
    dx = torch.linalg.lstsq(H + reg * I, -g).solution
    return dx

def _line_search(
    energy_f: callable,
    z: Tensor,
    direction: Tensor,
    grad: Tensor,
    opts: NewtonOptions,
    initial_step: float = 1.0,
) -> float:
    """
    Simple Armijo backtracking line search (dense, scalar step).
    """
    t = float(initial_step)
    f0 = energy_f(z).detach()
    gTd = (grad @ direction).detach()

    alpha = float(opts.line_search_alpha)
    beta = float(opts.line_search_beta)

    can_break = False
    for _ in range(opts.line_search_max_steps):
        z_new = z + t * direction
        f_new = energy_f(z_new).detach()
        if f_new <= f0 + alpha * t * gTd:
            if can_break:
                return t
            can_break = True
            t = t / beta  # try a bit larger
        else:
            t = t * beta  # shrink

    return t


class BatchedSimplicitsSolver:
    """
    API-compatible with your existing driver (compare_solvers.py).

    Important stored tensors (in reduced DOFs z):
      - B_sim: (3*num_qp, num_dof)  maps z -> dx_flat
      - dFdz:  (9*num_qp, num_dof)  maps z -> dF_flat (no +I)
      - bigI:  (9*num_qp,)          flattened identity per qp
      - M:     (num_dof, num_dof)   reduced mass BMB (dense)
      - vols:  (num_qp,)
      - x0_flat: (3*num_qp,)        sim_pts flattened
      - Bq:    (num_qp,3,num_dof)   view of B_sim for per-qp ops
      - dFq:   (num_qp,9,num_dof)   view of dFdz for per-qp ops
    """

    def __init__(
        self,
        scene,
        B_full_dense: Tensor,
        rho_init: Union[float, Tensor],
        device: str = "cuda",
        floor_height: float = 0.0,
        gravity_acc: float = -9.8,
        floor_axis: int = 2,
    ):
        self.device = device
        self.scene = scene

        # Core sim matrices from scene (warp BSR -> dense torch)
        with torch.no_grad():
            self.B_sim = _bsr_to_dense(scene.sim_B, device)       # (3N, ndof)
            self.dFdz = _bsr_to_dense(scene.sim_dFdz, device)     # (9N, ndof)
            self.BMB = _bsr_to_dense(scene.sim_BMB, device)       # (ndof, ndof) for the scene's current rho

            self.vols = wp.to_torch(scene.sim_vols).to(device).to(torch.float32)
            sim_pts = wp.to_torch(scene.sim_pts).to(device).to(torch.float32)
            self.x0_flat = sim_pts.reshape(-1)  # (3N,)

        # Shapes
        self.num_qp = self.vols.numel()
        self.num_dof = self.B_sim.shape[1]

        # Helpful views for per-qp assembly
        self.Bq = self.B_sim.view(self.num_qp, 3, self.num_dof)          # (N,3,ndof)
        self.dFq = self.dFdz.view(self.num_qp, 9, self.num_dof)          # (N,9,ndof)

        # bigI matches easy_api_torch: tiled 3x3 identity flattened per qp
        eye9 = torch.eye(3, device=device, dtype=torch.float32).reshape(-1)  # (9,)
        self.bigI = eye9.repeat(self.num_qp)  # (9N,)

        # Store full-space B only if needed externally (not used in Newton)
        self.B_full = B_full_dense.to(device)

        # Floor & gravity
        self.floor_height = float(floor_height)
        self.floor_axis = int(floor_axis)
        self.gravity_vec = torch.tensor([0.0, 0.0, float(gravity_acc)], device=device, dtype=torch.float32)
        self.gravity_dir = float(torch.sign(self.gravity_vec[self.floor_axis]).item())


        print("GRAVITY", self.gravity_vec, "gravity_dir", self.gravity_dir, "floor_height", self.floor_height, "floor_axis", self.floor_axis)
        # Density scaling:
        # Kaolin's scene.sim_BMB was built using the scene's rho field at creation/training time.
        # Your compare script passes a single uniform rho. To support rho optimization, we
        # factorize BMB into M_base * rho, assuming linearity in uniform rho:
        rho0 = float(rho_init.item()) if isinstance(rho_init, torch.Tensor) else float(rho_init)
        self.M_base = self.BMB / max(rho0, 1e-8)
        self.current_rho = (rho_init.clone() if isinstance(rho_init, torch.Tensor)
                            else torch.tensor(rho0, device=device, dtype=torch.float32))
        self.M = self.M_base * self.current_rho  # (ndof, ndof)

        # Default timestep cache (driver passes dt explicitly too)
        self.frame_dt = float(scene.timestep)

    def get_full_points_batch(self, structure_points, z_batch, skinning_weight_function):
        return weight_function_lbs(structure_points, z_batch.reshape(-1,3,4).unsqueeze(0), skinning_weight_function).squeeze()

    def set_density(self, rho_val: Union[float, Tensor]) -> None:
        if isinstance(rho_val, torch.Tensor):
            self.current_rho = rho_val
            self.M = self.M_base * rho_val
        else:
            rho_val = float(rho_val)
            self.current_rho = torch.tensor(rho_val, device=self.device, dtype=torch.float32)
            self.M = self.M_base * self.current_rho

    # ----------------------------
    # Elastic (Neo-Hookean) in z
    # ----------------------------
    def _elastic_energy_grad_hess(self, z: Tensor, mu: Tensor, lam: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns (E_elastic, g_elastic, H_elastic) in z-space.

        Uses Kaolin's legacy torch formulas for Neo-Hookean w.r.t. deformation gradients,
        then maps through dFdz (exactly like Easy API does via wrappers).
        """
        # F_flat = dFdz @ z + bigI   (9N,)
        F_flat = (self.dFdz @ z) + self.bigI
        F_flat_col = F_flat.reshape(-1, 1)  # legacy functions expect (9N,1)

        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, device=z.device, dtype=z.dtype).reshape(-1, 1)
        if not torch.is_tensor(lam):
            lam = torch.tensor(lam, device=z.device, dtype=z.dtype).reshape(-1, 1)
        # mu, lam as shape (N,1) for unbatched_* functions

        if mu.shape[0] != self.num_qp:
            mu_t = mu.reshape(1, 1).expand(self.num_qp, 1)
            
        else:
            mu_t = mu.reshape(self.num_qp, 1)

        if lam.shape[0] != self.num_qp:
            lam_t = lam.reshape(1, 1).expand(self.num_qp, 1)
        else:
            lam_t = lam.reshape(self.num_qp, 1)

        # Energy per primitive (N,1), gradient (N,3,3), hessian (N,9,9)
        e_per = unbatched_neohookean_energy(mu_t, lam_t, F_flat_col).squeeze(1)  # (N,)
        E = torch.sum(e_per * self.vols)

        F_3x3 = F_flat.view(self.num_qp, 3, 3)
        J = torch.linalg.det(F_3x3)
        
        # If any point is inverted (J <= 0.05), force Energy to Infinity.
        # This signals the Line Search to BACKTRACK immediately.
        if (J < 0.05).any():
            E = torch.tensor(float('inf'), device=z.device, dtype=z.dtype)

        # Gradient wrt F: (N,3,3). We flatten to (N,9) for mapping.
        gF = unbatched_neohookean_gradient(mu_t, lam_t, F_flat_col)  # (N,3,3)
        gF_flat = gF.reshape(self.num_qp, 9) * self.vols[:, None]  # (N,9)
        gF_vec = gF_flat.reshape(-1)  # (9N,)

        # Map to z-space: g = dFdz^T @ gF_vec
        g = self.dFdz.T @ gF_vec  # (ndof,)

        # Hessian: H = sum_q dFq^T (H_F_q * vol_q) dFq
        HF = unbatched_neohookean_hessian(mu_t, lam_t, F_flat_col)  # (N,9,9)
        HF = HF * self.vols[:, None, None]  # volume scaling

        # temp_q = HF_q @ dFq  -> (N,9,ndof)
        temp = torch.einsum("qij,qjk->qik", HF, self.dFq)
        # H = sum_q dFq^T @ temp
        H = torch.einsum("qia,qib->ab", self.dFq, temp)

        return E, g, H

    # ----------------------------
    # Point-wise forces in z
    # ----------------------------
    def _world_positions(self, z: Tensor) -> Tensor:
        """
        x = B_sim @ z + x0_flat, reshaped to (N,3)
        """
        x_flat = (self.B_sim @ z) + self.x0_flat
        return x_flat.view(self.num_qp, 3)

    def _gravity_energy_grad(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        E = rho * sum(vol * dot(g, x)), grad_z = rho * B^T @ (vol * g repeated)
        """
        x = self._world_positions(z)
        dot = torch.sum(x * self.gravity_vec[None, :], dim=1)  # (N,)
        E = self.current_rho * torch.sum(self.vols * dot)

        # g_x_flat: (3N,) where each point contributes vol*g
        gx = (self.vols[:, None] * self.gravity_vec[None, :]).reshape(-1)  # (3N,)
        g = self.current_rho * (self.B_sim.T @ gx)
        return E, g

    def _floor_energy_grad_hess(self, z: Tensor, floor_penalty: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Floor penalty in Kaolin:
          E = k * sum( vol * pen^2 ), pen = clamp(signed_dist, min=0)
        Hessian is nonzero only along the floor_axis for penetrating points.
        """
        x = self._world_positions(z)
        coord = x[:, self.floor_axis]  # (N,)

        if self.gravity_dir < 0:
            # gravity points +axis, "floor" is an upper plane
            pen = torch.clamp(coord - self.floor_height, min=0.0)
            sign = 1.0
        else:
            pen = torch.clamp(self.floor_height - coord, min=0.0)
            sign = -1.0  # d(pen)/d(coord) = -1 when active

        active = (pen > 0).to(z.dtype)  # (N,)
        k = float(floor_penalty)

        E = k * torch.sum((pen ** 2))

        # grad_x on axis: dE/dcoord = 2*k*vol*pen*sign when active
        gcoord = (2.0 * k) *  pen * sign * active  # (N,)

        # Map to z: g = sum_q Bq[q,axis,:]^T * gcoord[q]
        Ba = self.Bq[:, self.floor_axis, :]  # (N, ndof)
        g = torch.einsum("qn,q->n", Ba, gcoord)

        # Hessian in x: d2E/dcoord2 = 2*k*vol when active, else 0
        hcoord = (2.0 * k) * active  # (N,)
        # H = sum_q hcoord[q] * (Ba[q]^T Ba[q])
        # Compute efficiently: (Ba * sqrt(hcoord))^T (Ba * sqrt(hcoord))
        w = torch.sqrt(torch.clamp(hcoord, min=0.0)).unsqueeze(1)  # (N,1)
        A = Ba * w  # (N, ndof)
        H = A.T @ A  # (ndof, ndof)

        return E, g, H

    def _boundary_energy_grad_hess(
        self,
        z: Tensor,
        penalty_data_list: List[Tuple[Tensor, Tensor, Union[float, Tensor]]],
        batch_index: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        penalty_data_list: list of (indices, target_pos_batch, k_pen)
          - indices: (K,) long/int tensor indexing into cubature points
          - target_pos_batch: either (B,K,3) or (K,3)
          - k_pen: float or tensor (B,) or scalar tensor

        Energy per Kaolin boundary kernel:
          E = coeff * sum_i ||x_i - pinned_i||^2
        """
        if not penalty_data_list:
            return (torch.tensor(0.0, device=z.device, dtype=z.dtype),
                    torch.zeros(self.num_dof, device=z.device, dtype=z.dtype),
                    torch.zeros(self.num_dof, self.num_dof, device=z.device, dtype=z.dtype))

        x = self._world_positions(z)

        E_tot = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        g_tot = torch.zeros(self.num_dof, device=z.device, dtype=z.dtype)
        H_tot = torch.zeros(self.num_dof, self.num_dof, device=z.device, dtype=z.dtype)

        for (indices, target_pos_batch, k_pen) in penalty_data_list:
            # Targets
            if target_pos_batch.ndim == 3:
                tgt = target_pos_batch[batch_index]  # (K,3)
            else:
                tgt = target_pos_batch  # (K,3)

            # Coeff
            if isinstance(k_pen, torch.Tensor):
                coeff = k_pen[batch_index] if k_pen.ndim >= 1 else k_pen
            else:
                coeff = torch.tensor(float(k_pen), device=z.device, dtype=z.dtype)

            idx = indices.long()
            curr = x[idx, :]  # (K,3)
            diff = curr - tgt  # (K,3)

            # Energy
            E = coeff * torch.sum(diff * diff)
            E_tot = E_tot + E

            # Gradient in x: 2*coeff*diff
            g_x = (2.0 * coeff) * diff  # (K,3)

            # Map to z:
            # g = sum_i B_i^T g_x_i, with B_i = Bq[idx_i,:,:]
            Bi = self.Bq[idx, :, :]  # (K,3,ndof)
            g = torch.einsum("kcn,kc->n", Bi, g_x)
            g_tot = g_tot + g

            # Hessian in x: 2*coeff*I3 per constrained point
            # Map: H += sum_i 2*coeff * (B_i^T I B_i) = 2*coeff * sum_i (B_i^T B_i)
            # Compute: sum_i (Bi^T Bi)
            # Bi: (K,3,ndof) -> reshape (K*3, ndof), then (B^T B)
            Bflat = Bi.reshape(-1, self.num_dof)  # (3K, ndof)
            H = (2.0 * coeff) * (Bflat.T @ Bflat)
            H_tot = H_tot + H

        return E_tot, g_tot, H_tot

    # ----------------------------
    # Public API
    # ----------------------------
    def newton_step_batch(
        self,
        z_prev: Tensor,                 # (B, ndof)
        v_prev: Tensor,                 # (B, ndof)
        mu: Tensor,
        lam: Tensor,
        penalty_data_list: List[Tuple[Tensor, Tensor, Union[float, Tensor]]],
        dt: float,
        floor_penalty: float = 1.0,
        num_iters: int = 10,
        friction_coeff: float = 0.0,    # accepted; friction not modeled in this solver
        P: Optional[Tensor] = None,      # accepted for compatibility; currently unused
        Pt: Optional[Tensor] = None,     # accepted for compatibility; currently unused
        more_partial_newton_E=None,      # accepted for compatibility; ignored
        more_partial_newton_G=None,      # accepted for compatibility; ignored
        more_partial_newton_H=None,      # accepted for compatibility; ignored
    ) -> Tensor:
        """
        Runs Newton solve per batch element, returning z_new (B, ndof).
        """
        B = z_prev.shape[0]
        z_out = []

        # Newton options (use provided num_iters to mirror caller)
        opts = NewtonOptions(max_iters=int(num_iters))

        dt = float(dt)

        for b in range(B):
            z0 = z_prev[b]
            z_dot = v_prev[b]

            # Kaolin initializes Newton from explicit Euler guess (common in Easy API)
            z = (z0 + dt * z_dot).clone()

            # Precompute inertial linear terms
            M = self.M

            # Objective pieces in z-space
            def energy(zk: Tensor) -> Tensor:
                # Inertial (Kaolin _newton_E form, constant omitted)
                inertial = 0.5 * (zk @ (M @ zk)) - (zk @ (M @ z0)) - dt * (zk @ (M @ z_dot))

                # Potential: elastic + gravity + floor + boundary
                e_el, _, _ = self._elastic_energy_grad_hess(zk, mu, lam)
                e_g, _ = self._gravity_energy_grad(zk)
                e_f, _, _ = self._floor_energy_grad_hess(zk, floor_penalty)
                e_b, _, _ = self._boundary_energy_grad_hess(zk, penalty_data_list, b)

                return inertial + (dt * dt) * (e_el + e_g + e_f + e_b)

            for _ in range(opts.max_iters):
                # Assemble gradient/Hessian analytically
                # Inertial parts:
                g_in = (M @ z) - (M @ z0) - dt * (M @ z_dot)
                H_in = M

                # Potential parts:
                e_el, g_el, H_el = self._elastic_energy_grad_hess(z, mu, lam)
                e_g, g_g = self._gravity_energy_grad(z)
                e_f, g_f, H_f = self._floor_energy_grad_hess(z, floor_penalty)
                e_b, g_b, H_b = self._boundary_energy_grad_hess(z, penalty_data_list, b)

                newton_G = g_in + (dt * dt) * (g_el + g_g + g_f + g_b)
                newton_H = H_in + (dt * dt) * (H_el + H_f + H_b)  # gravity has 0 Hessian

                # Newton direction
                newton_H = newton_H + 1e-4 * torch.eye(newton_H.shape[0], device=z.device)
                
                try:
                    p = -torch.linalg.solve(newton_H, newton_G)
                except Exception as e:
                    print("error in solving", e) 
                    p = -1e-5 * newton_G
                # Kaolin convergence check: abs(p^T g) < conv_tol
                if torch.abs(p @ newton_G) < opts.conv_tol:
                    break

                # Line search
                step = _line_search(energy, z, p, newton_G, opts, initial_step=1.0)
                if step < 1e-8:
                    # If step collapses, accept a tiny step to avoid stalling.
                    step = 1e-8

                z = z + step * p

            z_out.append(z)

        return torch.stack(z_out, dim=0)
