#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3: Data sources and scatterer shapes.

Layout: 5 rows (sphere, cube, two_spheres+cube, dispersed, inhomogeneous) ×
         4 cols (truth, Full VIE, VIE-Born FF, Analytic Born FF).
Truncation: GPSWF params (ell_max=12, n_modes=7) + epsilon 0.1 + N_cap.
"""
from __future__ import annotations

import argparse, math; from pathlib import Path; from typing import Any
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
from scipy.linalg import lu_factor, lu_solve

from common import (
    collect_reconstruction_diagnostics,
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    generate_data_nodes, make_table, modal_matrix, orthonormal_basis_perp,
    plot_diagnostic_curves,
    quadrature_modal_coefficients, recover_polarimetric_coefficients,
    reference_tensor, save_diagnostics_npz,
    solve_ball_gpswf, sphere_quadrature, tensor_coefficients_from_matrix,
    write_diagnostics_csv,
)
from common.phantom import (
    Block, Mode,
    _shape_truth_and_fourier,
    cube_phantom, two_spheres_cube_phantom, dispersed_blocks_phantom,
)
from forward.datasets import (
    FarfieldDataset, analytic_born_farfield_dataset,
    discrete_vie_born_farfield_dataset, full_vie_farfield_dataset,
    farfield_dataset_to_qhat,
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
        return 26, 10, 170, 51, 5, 30, 8, 5, 60, 50
    return 74, 12, 230, 81, 19, 50, 12, 7, 140, 100


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, requested_target_dirs, grid_size,
     n_per_axis, K, ell_max, n_modes_per_ell, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k; R = 1.0; epsilon = 0.1
    kind = "full"; component_index = 0
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
    alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
    retained = alpha_abs > epsilon * float(np.max(alpha_abs))
    N_cap = int(C * C / 2)
    if np.sum(retained) > N_cap:
        order = np.argsort(-alpha_abs)
        keep = order[:N_cap]
        retained = np.zeros(len(modes), dtype=bool); retained[keep] = True

    # -- Plot --
    n_rows = len(SHAPES); n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    diagnostic_rows: list[dict[str, Any]] = []

    for row_idx, shape_name in enumerate(SHAPES):
        print(f"  Processing {shape_name}...")
        # Analytic Born far-field data uses continuous phantom Fourier formulas.
        fourier_nodes = target_nodes if data_mode == 'ideal' else p_nodes
        truth, gps, dm, analytic_born_farfield = _shape_truth_and_fourier(
            shape_name, fourier_nodes, grid_size, C)
        image_matrix = modal_matrix(gps, modes, fourier_side=False)
        vmin = float(np.nanmin(np.real(truth))); vmax = float(np.nanmax(np.real(truth)))

        # --- Analytic Born far-field (column 4): full pipeline ---
        coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)
        tc_ana = analytic_born_farfield[:, None] * coeff0[None, :]
        rec_c_ana, _, _ = recover_polarimetric_coefficients(
            p_nodes, tc_ana, kind, 0.0, rng)
        comp_data_ana = rec_c_ana[:, component_index]
        coeffs_ana = quadrature_modal_coefficients(
            comp_data_ana, target_basis, target_weights, modes, retained)
        rec_ana = (image_matrix @ coeffs_ana).reshape(grid_size, grid_size)
        rec_ana[~dm] = 0.0
        effective_p_nodes = target_nodes if data_mode == 'ideal' else p_nodes
        effective_mock_distances = np.zeros(target_nodes.shape[0]) if data_mode == 'ideal' else mock_distances
        base_case = {
            "figure": 3,
            "shape_name": shape_name,
            "k": float(k),
            "C": float(C),
            "K": int(K),
            "ell_max": int(ell_max),
            "n_modes_per_ell": int(n_modes_per_ell),
            "epsilon": float(epsilon),
            "N_cap": int(N_cap),
            "n_radial": int(n_radial),
            "n_angular_requested": int(requested_target_dirs),
            "data_mode": data_mode,
            "noise_level": 0.0,
            "contrast_scale": math.nan,
        }
        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                **base_case,
                "case_id": f"{shape_name}_analytic_born_farfield",
                "row": int(row_idx),
                "column": 3,
                "data_source": "analytic_born_farfield",
            },
            modes=modes,
            retained=retained,
            target_nodes=target_nodes,
            p_nodes=effective_p_nodes,
            mock_distances=effective_mock_distances,
            basis_matrix=target_basis,
            target_weights=target_weights,
            component_data=comp_data_ana,
            coeffs=coeffs_ana,
            image=rec_ana,
            truth=truth,
            disk_mask=dm,
        ))

        # --- VIE data (columns 2, 3): unified farfield → polarimetric → GPSWF ---
        vie_nodes = target_nodes if data_mode == 'ideal' else p_nodes
        # Full VIE
        ds_full = full_vie_farfield_dataset(
            shape_name, vie_nodes, kind=kind, k=k, R=R,
            n_per_axis=n_per_axis, n_geometries=6)
        rec_c_full = farfield_dataset_to_qhat(ds_full, kind=kind, noise_level=0.0, rng=rng)
        comp_full = rec_c_full[:, component_index]
        if data_mode == 'ideal':
            comp_full = comp_full.reshape(-1, target_nodes.shape[0]).mean(axis=0)
        coeffs_full = quadrature_modal_coefficients(
            comp_full, target_basis, target_weights, modes, retained)
        rec_full = (image_matrix @ coeffs_full).reshape(grid_size, grid_size)
        rec_full[~dm] = 0.0

        # VIE Born
        ds_born = discrete_vie_born_farfield_dataset(
            shape_name, vie_nodes, kind=kind, k=k, R=R,
            n_per_axis=n_per_axis, n_geometries=6)
        rec_c_born = farfield_dataset_to_qhat(ds_born, kind=kind, noise_level=0.0, rng=rng)
        comp_born = rec_c_born[:, component_index]
        if data_mode == 'ideal':
            comp_born = comp_born.reshape(-1, target_nodes.shape[0]).mean(axis=0)
        coeffs_bv = quadrature_modal_coefficients(
            comp_born, target_basis, target_weights, modes, retained)
        rec_bv = (image_matrix @ coeffs_bv).reshape(grid_size, grid_size)
        rec_bv[~dm] = 0.0

        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                **base_case,
                "case_id": f"{shape_name}_full_vie",
                "row": int(row_idx), "column": 1, "data_source": "full_vie",
            },
            modes=modes, retained=retained,
            target_nodes=target_nodes, p_nodes=effective_p_nodes,
            mock_distances=effective_mock_distances,
            basis_matrix=target_basis, target_weights=target_weights,
            component_data=comp_full, coeffs=coeffs_full,
            image=rec_full, truth=truth, disk_mask=dm,
        ))
        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                **base_case,
                "case_id": f"{shape_name}_vie_born",
                "row": int(row_idx), "column": 2, "data_source": "discrete_vie_born_farfield",
            },
            modes=modes, retained=retained,
            target_nodes=target_nodes, p_nodes=effective_p_nodes,
            mock_distances=effective_mock_distances,
            basis_matrix=target_basis, target_weights=target_weights,
            component_data=comp_born, coeffs=coeffs_bv,
            image=rec_bv, truth=truth, disk_mask=dm,
        ))

        # --- Plot row (shared vmin/vmax from truth, same style as Figure 1) ---
        titles = ["truth", "Full VIE", "VIE-Born FF", "Analytic Born FF"] if row_idx == 0 else [""]*4
        images = [np.real(truth), np.real(rec_full), np.real(rec_bv), np.real(rec_ana)]
        for col_idx, (img, title) in enumerate(zip(images, titles)):
            _imshow(axes[row_idx, col_idx], img, title, "viridis", vmin, vmax)

        axes[row_idx, 0].set_ylabel(shape_name, fontsize=10, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure3_sources_shapes.png", dpi=200)
    plt.close(fig)
    write_diagnostics_csv(diagnostic_rows, config.out_dir / "figure3_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "figure3_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "figure3_diagnostic_curves.png",
        title="Figure 3 diagnostics",
    )
    print(f"Saved {config.out_dir / 'figure3_sources_shapes.png'}")
    return make_table([{"figure": 3, "status": "ok"}])


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
