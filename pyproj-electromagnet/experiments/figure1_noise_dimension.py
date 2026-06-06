#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1: Noise and truncation dimension effects (Born data, k=15).

Layout: 4 rows (N=5,72,144,256) × 5 cols (truth + noise=0,0.1,0.2,0.3).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    farfield_fourier_nodes, interior_ball_nodes, make_table,
    match_mock_quadrature_nodes, modal_matrix, print_table,
    quadrature_modal_coefficients, recover_polarimetric_coefficients,
    reference_tensor, solve_ball_gpswf, sphere_quadrature,
    tensor_coefficients_from_matrix, vector_norm,
)
from common.phantom import Block, Mode, three_block_phantom, block_fourier_profile, truth_image_2d


def _settings(quick: bool):
    if quick:
        return 38, 6, 110, 51, 40, 10, 6, [5, 20, 40, 60], 100, 80
    # Full settings
    return 110, 12, 230, 81, 60, 18, 10, [5, 72, 144, 256], 160, 120


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, requested_target_dirs, grid_size,
     K, ell_max, n_modes_per_ell, N_values, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k; kind = "full"; component_index = 0
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    rng = np.random.default_rng(config.seed + 100)

    # -- Setup nodes and data --
    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, requested_target_dirs)
    directions, _, _ = sphere_quadrature(requested_measure_dirs, "lebedev")
    available_nodes = interior_ball_nodes(farfield_fourier_nodes(directions, directions))
    indices, _ = match_mock_quadrature_nodes(target_nodes, available_nodes)
    p_nodes = available_nodes[indices]

    blocks = three_block_phantom("born")
    scalar = block_fourier_profile(p_nodes, blocks, C=C)
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)
    true_coeffs = scalar[:, None] * coeff0[None, :]
    data_clean = true_coeffs[:, component_index]

    truth, grid_points, disk_mask = truth_image_2d(grid_size, blocks, coeff0[component_index])

    # -- Build GPSWF basis --
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
            alpha = alpha_lookup[(ell, n)]
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=alpha, beta=beta[:, n]))

    target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
    image_matrix = modal_matrix(grid_points, modes, fourier_side=False)
    alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
    alpha_max = float(np.max(alpha_abs))
    order = np.argsort(-alpha_abs)

    # -- Plot --
    fig, axes = plt.subplots(len(N_values), 1 + len(noise_levels),
                             figsize=(3.1 * (1 + len(noise_levels)), 3.1 * len(N_values)),
                             constrained_layout=True)
    if len(N_values) == 1:
        axes = axes[None, :]

    vmin = float(np.nanmin(np.real(truth))); vmax = float(np.nanmax(np.real(truth)))

    for row_idx, N in enumerate(N_values):
        dim = min(N, len(modes))
        selected = order[:dim]
        retained = np.zeros(len(modes), dtype=bool); retained[selected] = True

        # Truth (first column, only first row gets label)
        if row_idx == 0:
            _imshow(axes[row_idx, 0], np.real(truth), f"ground truth", "viridis", vmin, vmax)
        else:
            _imshow(axes[row_idx, 0], np.real(truth), "", "viridis", vmin, vmax)

        for col_idx, noise_level in enumerate(noise_levels):
            rec_coeffs, _, _ = recover_polarimetric_coefficients(
                p_nodes, true_coeffs, kind, noise_level, rng)
            comp_data = rec_coeffs[:, component_index]
            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
            rec[~disk_mask] = 0.0
            label = f"N={dim}, δ={noise_level:g}" if row_idx == 0 else ""
            _imshow(axes[row_idx, 1 + col_idx], np.real(rec), label, "viridis", vmin, vmax)

    # Row labels
    for row_idx, N in enumerate(N_values):
        axes[row_idx, 0].set_ylabel(f"N={min(N, len(modes))}", fontsize=10, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure1_noise_dimension.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.out_dir / 'figure1_noise_dimension.png'}")
    return make_table([{"figure": 1, "status": "ok"}])


def _imshow(ax, img, title, cmap, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


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
