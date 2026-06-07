#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2: Frequency and contrast effects (Born data, noise=0.2).

Layout: 4 rows (k=10,20,30,40) × 4 cols (truth + low/medium/high contrast).
Truncation: article-style — (ell_max, n_modes_per_ell, K, n_radial, n_angular)
linked to C; all generated GPSWF modes are retained.
"""
from __future__ import annotations

import argparse; from pathlib import Path; from typing import Any
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np

from common import (
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    generate_data_nodes, make_table, modal_matrix,
    quadrature_modal_coefficients, recover_polarimetric_coefficients,
    reference_tensor, solve_ball_gpswf,
    tensor_coefficients_from_matrix,
)
from common.phantom import Block, Mode, three_block_phantom, block_fourier_profile, truth_image_2d


def _row_params(k: float) -> dict:
    """Article-style GPSWF parameters linked to wave number k."""
    C = 2.0 * k
    if k <= 10:
        return {"ell_max": 8,  "n_modes": 5,  "K": 36, "n_radial": 8,  "n_angular": 110, "C": C}
    elif k <= 20:
        return {"ell_max": 14, "n_modes": 6,  "K": 54, "n_radial": 12, "n_angular": 170, "C": C}
    elif k <= 30:
        return {"ell_max": 18, "n_modes": 7,  "K": 68, "n_radial": 14, "n_angular": 230, "C": C}
    else:
        return {"ell_max": 22, "n_modes": 8,  "K": 86, "n_radial": 16, "n_angular": 302, "C": C}


def _row_params_quick(k: float) -> dict:
    C = 2.0 * k
    if k <= 12:
        return {"ell_max": 6, "n_modes": 4, "K": 24, "n_radial": 5, "n_angular": 74,  "C": C}
    else:
        return {"ell_max": 8, "n_modes": 4, "K": 30, "n_radial": 6, "n_angular": 110, "C": C}


def run_experiment(config: ExperimentConfig) -> Any:
    requested_measure_dirs = 110; grid_size = 81
    noise_level = 0.2; kind = "full"; component_index = 0
    contrast_scales = [0.3, 1.0, 3.0]
    quad_order = 160; r_eval_count = 120
    k_values = [10, 20, 30, 40]
    n_cols = 1 + 1 + len(contrast_scales)  # truth + noiseless_medium + low/med/high

    if config.quick:
        requested_measure_dirs = 38; grid_size = 51
        k_values = [10, 15]
        quad_order = 100; r_eval_count = 80

    rng = np.random.default_rng(config.seed + 200)
    data_mode = getattr(config, 'data_mode', 'mock')
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows = len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    if n_rows == 1: axes = axes[None, :]

    for row_idx, k in enumerate(k_values):
        rp = _row_params_quick(k) if config.quick else _row_params(k)
        C = rp["C"]; ell_max = rp["ell_max"]; n_modes = rp["n_modes"]
        K_val = rp["K"]; n_radial = rp["n_radial"]; n_angular = rp["n_angular"]

        # Target quadrature for this k
        target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)
        p_nodes, _, _, _, _ = generate_data_nodes(
            target_nodes, requested_measure_dirs, data_mode=data_mode)

        # GPSWF basis — all modes retained
        alpha_df = collect_alpha_pairs_cached(
            C, K_val, ell_max, n_modes, quad_order=quad_order,
            r_eval_count=r_eval_count, cache_dir=config.out_dir / "alpha_cache")
        alpha_lookup = {(int(r["ell"]), int(r["n"])): complex(float(r["alpha_real"]), float(r["alpha_imag"]))
                        for _, r in alpha_df.iterrows()}
        modes = []
        for ell in range(ell_max + 1):
            _, beta = solve_ball_gpswf(C, ell, K_val, n_modes=n_modes)
            for n in range(beta.shape[1]):
                a = alpha_lookup[(ell, n)]
                for m in range(-ell, ell + 1):
                    modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))
        # Article-style: all generated modes, minus tiny-alpha tail
        alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
        retained = alpha_abs > 0.2 * float(np.max(alpha_abs))

        target_basis = modal_matrix(target_nodes, modes, fourier_side=True)

        # Grid
        xs = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(xs, xs)
        gps = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
        image_matrix = modal_matrix(gps, modes, fourier_side=False)

        blocks = three_block_phantom("born")
        truth, _, dm = truth_image_2d(grid_size, blocks, coeff0[component_index])
        vmin = float(np.nanmin(np.real(truth))); vmax = float(np.nanmax(np.real(truth)))

        contrast_labels = {0.3: "low", 1.0: "medium", 3.0: "high"}
        columns = [(None, None, "truth")]  # (scale, noise, label)
        columns.append((1.0, 0.0, f"k={k}, medium δ=0"))
        for scale in contrast_scales:
            columns.append((scale, noise_level, f"k={k}, {contrast_labels[scale]}"))

        for col_idx, (scale, nlevel, label) in enumerate(columns):
            if scale is None:
                _imshow(axes[row_idx, col_idx], np.real(truth),
                        "truth" if row_idx == 0 else "", "viridis", vmin, vmax)
                continue

            scaled_blocks = [Block(center=b.center, half_width=b.half_width,
                amplitude=complex(b.amplitude.real * scale, b.amplitude.imag * scale)) for b in blocks]
            scalar = block_fourier_profile(p_nodes, scaled_blocks, C=C)
            tc = scalar[:, None] * coeff0[None, :]
            rec_c, _, _ = recover_polarimetric_coefficients(p_nodes, tc, kind, nlevel, rng)
            comp_data = rec_c[:, component_index]
            if data_mode == 'ideal':
                comp_data = comp_data.reshape(-1, target_nodes.shape[0]).mean(axis=0)
            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
            rec[~dm] = 0.0
            _imshow(axes[row_idx, col_idx], np.real(rec),
                    label if row_idx == 0 else "", "viridis", vmin, vmax)

        n_total = len(modes)
        axes[row_idx, 0].set_ylabel(f"k={k} ({n_total})", fontsize=10, rotation=90, labelpad=12)

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
