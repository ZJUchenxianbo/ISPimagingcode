#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2: Frequency and contrast effects (Born data, noise=0.2).

Layout: 4 rows (k=10,20,30,40) × 4 cols (truth + low/medium/high contrast).
Truncation: fixed N=256 modes sorted by |alpha|.
"""
from __future__ import annotations

import argparse; from pathlib import Path; from typing import Any
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np

from common import (
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    generate_data_nodes, make_table, modal_matrix,
    quadrature_modal_coefficients, recover_polarimetric_coefficients,
    reference_tensor, solve_ball_gpswf, sphere_quadrature,
    tensor_coefficients_from_matrix,
)
from common.phantom import Block, Mode, three_block_phantom, block_fourier_profile, truth_image_2d


def _settings(quick: bool):
    if quick:
        return 38, 6, 110, 51, [10, 15], [0.3, 1.0, 3.0], 40, 12, 8, 100, 80
    return 110, 12, 230, 81, [10, 20, 30, 40], [0.3, 1.0, 3.0], 60, 18, 10, 160, 120


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, requested_target_dirs, grid_size,
     k_values, contrast_scales, K_base, ell_max, n_modes_per_ell,
     quad_order, r_eval_count) = _settings(config.quick)

    noise_level = 0.2; kind = "full"; component_index = 0
    rng = np.random.default_rng(config.seed + 200)

    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, requested_target_dirs)
    data_mode = getattr(config, 'data_mode', 'mock')
    p_nodes, _, _, mock_distances, data_info = generate_data_nodes(
        target_nodes, requested_measure_dirs, data_mode=data_mode)

    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows, n_cols = len(k_values), 1 + len(contrast_scales)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    if n_rows == 1: axes = axes[None, :]

    for row_idx, k in enumerate(k_values):
        C = 2.0 * k; K = K_base + (row_idx * 10)

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
        # Epsilon truncation: |alpha| > eps_ratio * max|alpha|
        # Modes retained automatically scales with C (Shannon number)
        eps_ratio = 0.1
        retained = alpha_abs > eps_ratio * float(np.max(alpha_abs))

        # Grid points for image reconstruction
        xs = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(xs, xs)
        gps = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
        image_matrix = modal_matrix(gps, modes, fourier_side=False)

        # Truth and shared vmin/vmax (medium contrast = scale 1.0)
        blocks = three_block_phantom("born")
        truth, _, dm = truth_image_2d(grid_size, blocks, coeff0[component_index])
        vmin = float(np.nanmin(np.real(truth))); vmax = float(np.nanmax(np.real(truth)))

        for col_idx, scale in enumerate([None] + list(contrast_scales)):
            if scale is None:
                # Truth column
                _imshow(axes[row_idx, 0], np.real(truth),
                        "truth" if row_idx == 0 else "", "viridis", vmin, vmax)
                continue

            scaled_blocks = [
                Block(center=b.center, half_width=b.half_width,
                      amplitude=complex(b.amplitude.real * scale, b.amplitude.imag * scale))
                for b in blocks
            ]
            scalar = block_fourier_profile(p_nodes, scaled_blocks, C=C)
            tc = scalar[:, None] * coeff0[None, :]
            rec_c, _, _ = recover_polarimetric_coefficients(p_nodes, tc, kind, noise_level, rng)
            comp_data = rec_c[:, component_index]
            if data_mode == 'ideal':
                n_target = target_nodes.shape[0]
                comp_data = comp_data.reshape(-1, n_target).mean(axis=0)
            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
            rec[~dm] = 0.0
            cl = ["low", "medium", "high"][col_idx - 1]  # col_idx 1,2,3
            label = f"k={k}, {cl}" if row_idx == 0 else ""
            _imshow(axes[row_idx, col_idx], np.real(rec), label, "viridis", vmin, vmax)

        axes[row_idx, 0].set_ylabel(f"k={k}", fontsize=10, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure2_frequency_contrast.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.out_dir / 'figure2_frequency_contrast.png'}")
    return make_table([{"figure": 2, "status": "ok"}])


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
