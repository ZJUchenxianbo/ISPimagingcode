#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3: Data sources and scatterer shapes.

Layout: 4 rows (sphere, cube, two_spheres+cube, dispersed) ×
         4 cols (truth, Full VIE, VIE-Born, Analytical Born).
"""
from __future__ import annotations

import argparse, math; from pathlib import Path; from typing import Any
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
from scipy.linalg import lu_factor, lu_solve

from common import (
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    generate_data_nodes, make_table, modal_matrix, orthonormal_basis_perp,
    quadrature_modal_coefficients, reference_tensor,
    solve_ball_gpswf, sphere_quadrature, tensor_coefficients_from_matrix,
)
from common.phantom import (
    Block, Mode,
    _shape_truth_and_fourier,
    cube_phantom, two_spheres_cube_phantom, dispersed_blocks_phantom,
)
from forward.vie import (
    assemble_vie_matrix, ball_voxel_grid, incident_plane_wave,
    maxwell_born_far_field, maxwell_far_field, tensor_blocks_contrast,
    vie_to_fourier_convention,
)

SHAPES = ["sphere", "cube", "two_spheres_cube", "dispersed", "inhomogeneous"]


def _shape_to_blocks(name: str) -> list[Block]:
    if name == "sphere":
        return [Block(center=(0.0, 0.0, 0.0), half_width=(0.25, 0.25, 0.25), amplitude=1.0 + 0.0j)]
    elif name == "cube":
        return cube_phantom(center=(0.0, 0.0, 0.0), half_side=0.2, amplitude=1.0 + 0.0j)
    elif name == "two_spheres_cube":
        return two_spheres_cube_phantom()
    elif name == "dispersed":
        return dispersed_blocks_phantom()
    elif name == "inhomogeneous":
        return []  # Not block-based — built directly from Gaussian formula
    raise ValueError(name)


def _settings(quick: bool):
    if quick:
        return 26, 6, 86, 51, 5, 30, 8, 5, 60, 50
    return 74, 10, 170, 81, 11, 50, 12, 7, 140, 100


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, requested_target_dirs, grid_size,
     n_per_axis, K, ell_max, n_modes_per_ell, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k; R = 1.0; kind = "full"; component_index = 0
    rng = np.random.default_rng(config.seed + 300)

    # -- Setup --
    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, requested_target_dirs)
    data_mode = getattr(config, 'data_mode', 'mock')
    p_nodes, matched_inc, matched_obs, mock_distances, data_info = generate_data_nodes(
        target_nodes, requested_measure_dirs, data_mode=data_mode, branch_count=3)

    # -- GPSWF basis --
    alpha_df = collect_alpha_pairs_cached(
        C, K, ell_max, n_modes_per_ell, quad_order=quad_order,
        r_eval_count=r_eval_count, cache_dir=config.out_dir / "alpha_cache",
    )
    alpha_lookup = {(int(r["ell"]), int(r["n"])): complex(float(r["alpha_real"]), float(r["alpha_imag"]))
                    for _, r in alpha_df.iterrows()}
    modes = []
    for ell in range(ell_max + 1):
        _, beta = solve_ball_gpswf(C, ell, K, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            a = alpha_lookup[(ell, n)]
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))

    target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
    retained = np.ones(len(modes), dtype=bool)  # article-style: keep all

    # -- Voxel grid and tensor (isotropic for simplicity) --
    volume_nodes, volume_weights, voxel_h = ball_voxel_grid(R, n_per_axis)
    tensor = reference_tensor("isotropic")

    # -- Plot --
    n_rows = len(SHAPES); n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)

    for row_idx, shape_name in enumerate(SHAPES):
        print(f"  Processing {shape_name}...")
        # Analytical Fourier uses target nodes (not duplicated p_nodes)
        fourier_nodes = target_nodes if data_mode == 'ideal' else p_nodes
        truth, gps, dm, fourier_analytical = _shape_truth_and_fourier(
            shape_name, fourier_nodes, grid_size, C)
        image_matrix = modal_matrix(gps, modes, fourier_side=False)
        vmin = float(np.nanmin(np.real(truth))); vmax = float(np.nanmax(np.real(truth)))

        # --- Analytical Born (column 4) ---
        coeffs_ana = quadrature_modal_coefficients(
            fourier_analytical, target_basis, target_weights, modes, retained)
        rec_ana = (image_matrix @ coeffs_ana).reshape(grid_size, grid_size)
        rec_ana[~dm] = 0.0

        # --- VIE data (columns 2, 3) ---
        blocks = _shape_to_blocks(shape_name)
        if shape_name == "inhomogeneous":
            # Build Gaussian-bump contrast directly on voxel grid
            Q = np.zeros((volume_nodes.shape[0], 3, 3), dtype=np.complex128)
            bumps = [
                (np.array([-0.20, 0.15, 0.0]), 0.12, 1.0 + 0.0j),
                (np.array([0.25, 0.10, 0.0]), 0.08, 0.7 + 0.2j),
                (np.array([0.05, -0.25, 0.0]), 0.14, 1.2 - 0.1j),
            ]
            for center, sigma, amp in bumps:
                r_sq = np.sum((volume_nodes - center[None, :])**2, axis=1)
                scalar = complex(amp) * np.exp(-0.5 * r_sq / sigma**2)
                for a in range(3):
                    Q[:, a, a] += scalar  # isotropic: Q_aa = scalar
        else:
            Q = tensor_blocks_contrast(
                volume_nodes,
                [(np.asarray(b.center, dtype=float), np.asarray(b.half_width, dtype=float),
                  complex(b.amplitude)) for b in blocks],
                tensor)
        A = assemble_vie_matrix(volume_nodes, volume_weights, Q, k, h=voxel_h)
        lu = lu_factor(A)

        comp_full, comp_born_vie = _compute_vie_scalar_data(
            p_nodes, matched_inc, matched_obs,
            volume_nodes, volume_weights, Q, tensor, k, R, lu,
        )
        comp_full_u = vie_to_fourier_convention(comp_full)
        comp_born_vie_u = vie_to_fourier_convention(comp_born_vie)
        # In ideal mode, average over branches per target node
        if data_mode == 'ideal':
            n_target = target_nodes.shape[0]
            comp_full_u = comp_full_u.reshape(-1, n_target).mean(axis=0)
            comp_born_vie_u = comp_born_vie_u.reshape(-1, n_target).mean(axis=0)

        coeffs_full = quadrature_modal_coefficients(
            comp_full_u, target_basis, target_weights, modes, retained)
        coeffs_bv = quadrature_modal_coefficients(
            comp_born_vie_u, target_basis, target_weights, modes, retained)

        rec_full = (image_matrix @ coeffs_full).reshape(grid_size, grid_size); rec_full[~dm] = 0.0
        rec_bv = (image_matrix @ coeffs_bv).reshape(grid_size, grid_size); rec_bv[~dm] = 0.0

        # --- Plot row (shared vmin/vmax from truth, same style as Figure 1) ---
        titles = ["truth", "Full VIE", "VIE Born", "Analytical Born"] if row_idx == 0 else ["", "", "", ""]
        images = [np.real(truth), np.real(rec_full), np.real(rec_bv), np.real(rec_ana)]
        for col_idx, (img, title) in enumerate(zip(images, titles)):
            _imshow(axes[row_idx, col_idx], img, title, "viridis", vmin, vmax)

        axes[row_idx, 0].set_ylabel(shape_name, fontsize=10, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure3_sources_shapes.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.out_dir / 'figure3_sources_shapes.png'}")
    return make_table([{"figure": 3, "status": "ok"}])


def _compute_vie_scalar_data(p_nodes, inc_dirs, obs_dirs, v_nodes, v_weights, Q, tensor, k, R, lu):
    """Compute scalar Fourier data from VIE for all matched direction pairs."""
    coeff0 = tensor_coefficients_from_matrix(tensor, "isotropic")
    comp_scale = complex(coeff0[0])
    comp_full = np.zeros(p_nodes.shape[0], dtype=np.complex128)
    comp_born = np.zeros_like(comp_full)
    scale = (4.0 * math.pi / (float(k) ** 2)) / (float(R) ** 3)

    for idx in range(p_nodes.shape[0]):
        d = inc_dirs[idx]; xhat = obs_dirs[idx]
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        projector = np.eye(3) - np.outer(xhat, xhat)
        model_blocks, full_blocks, born_blocks = [], [], []
        for col in range(E_basis.shape[1]):
            e = E_basis[:, col].astype(np.complex128)
            rhs = incident_plane_wave(v_nodes, k, d, e).reshape(-1)
            total_field = lu_solve(lu, rhs).reshape((-1, 3))
            full = maxwell_far_field(v_nodes, v_weights, Q, total_field, k, xhat[None, :])[0]
            born = maxwell_born_far_field(v_nodes, v_weights, Q, k, d, e, xhat[None, :])[0]
            model_blocks.append(projector @ tensor @ e)
            full_blocks.append(scale * full)
            born_blocks.append(scale * born)
        model = np.concatenate(model_blocks)
        denom = np.vdot(model, model)
        if abs(denom) > 1e-14:
            comp_full[idx] = np.vdot(model, np.concatenate(full_blocks)) / denom * comp_scale
            comp_born[idx] = np.vdot(model, np.concatenate(born_blocks)) / denom * comp_scale
    return comp_full, comp_born


def _imshow(ax, img, title, cmap, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(out_dir=out_dir, seed=args.seed, quick=args.quick))


if __name__ == "__main__":
    main()
