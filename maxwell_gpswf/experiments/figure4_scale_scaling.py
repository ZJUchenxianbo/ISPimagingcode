#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4: Scale scaling — imaging region radius R varies, scatterer fixed.

Layout: 5 cols (truth + R=1.0/1.5/2.0/3.0) × 5 rows (shapes from fig3).
Data: Born analytical. Truncation: GPSWF params → ε=0.2 → N_cap=C²/2.
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
from common.phantom import (
    Block, Mode,
    _shape_truth_and_fourier,
)


def _row_params(R: float) -> dict:
    """GPSWF parameters linked to C = 2kR (k=15 fixed)."""
    C = 30.0 * R  # = 2 * 15 * R
    if R <= 1.0:
        return {"ell_max": 10, "n_modes": 5,  "K": 44, "n_radial": 12, "n_angular": 170, "C": C}
    elif R <= 1.5:
        return {"ell_max": 14, "n_modes": 6,  "K": 54, "n_radial": 14, "n_angular": 302, "C": C}
    elif R <= 2.0:
        return {"ell_max": 18, "n_modes": 7,  "K": 68, "n_radial": 16, "n_angular": 434, "C": C}
    else:
        return {"ell_max": 22, "n_modes": 8,  "K": 90, "n_radial": 20, "n_angular": 590, "C": C}


SHAPES = ["sphere", "cube", "two_spheres_cube", "dispersed", "inhomogeneous"]
R_VALUES = [1.0, 1.5, 2.0, 3.0]


def run_experiment(config: ExperimentConfig) -> Any:
    requested_measure_dirs = 110; grid_size = 81
    k = 15.0; epsilon = 0.2; kind = "full"; component_index = 0
    quad_order = 160; r_eval_count = 120

    rng = np.random.default_rng(config.seed + 400)
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows = len(SHAPES); n_cols = 1 + len(R_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)

    for row_idx, shape_name in enumerate(SHAPES):
        print(f"  Processing {shape_name}...")

        for col_idx, R in enumerate([None] + R_VALUES):
            if R is None:
                # Truth at R=1.0 (physical scale)
                rp0 = _row_params(1.0)
                tgt0, _, _ = ball_quadrature_nodes(rp0["n_radial"], rp0["n_angular"])
                truth_ref, _, dm_ref, _ = _shape_truth_and_fourier(
                    shape_name, tgt0, grid_size, rp0["C"])
                vmin_ref = float(np.nanmin(np.real(truth_ref)))
                vmax_ref = float(np.nanmax(np.real(truth_ref)))
                _imshow(axes[row_idx, 0], np.real(truth_ref),
                        "truth" if row_idx == 0 else "", "viridis",
                        vmin_ref, vmax_ref, extent=1.0)
                continue

            rp = _row_params(R); C = rp["C"]
            ell_max = rp["ell_max"]; n_modes = rp["n_modes"]; K_val = rp["K"]
            n_radial = rp["n_radial"]; n_angular = rp["n_angular"]

            # Quadrature
            target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)
            p_nodes, _, _, _, _ = generate_data_nodes(
                target_nodes, requested_measure_dirs, data_mode="mock", branch_count=1)

            # GPSWF
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

            alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
            retained = alpha_abs > epsilon * float(np.max(alpha_abs))
            N_cap = int(C * C / 2)
            if np.sum(retained) > N_cap:
                order = np.argsort(-alpha_abs)
                keep = order[:N_cap]
                retained = np.zeros(len(modes), dtype=bool); retained[keep] = True

            target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
            xs = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(xs, xs)
            gps = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
            image_matrix = modal_matrix(gps, modes, fourier_side=False)

            # Analytical Born data
            _, _, _, fourier_data = _shape_truth_and_fourier(shape_name, p_nodes, grid_size, C)
            rec_c, _, _ = recover_polarimetric_coefficients(
                p_nodes, fourier_data[:, None] * coeff0[None, :], kind, 0.0, rng)
            comp_data = rec_c[:, component_index]
            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)

            label = f"R={R}" if row_idx == 0 else ""
            rvmin = float(np.nanmin(np.real(rec))) if rec.size > 0 else -1
            rvmax = float(np.nanmax(np.real(rec))) if rec.size > 0 else 1
            if abs(rvmax - rvmin) < 1e-12: rvmin, rvmax = -1, 1
            _imshow(axes[row_idx, col_idx], np.real(rec), label, "viridis",
                    rvmin, rvmax, extent=R)

        n_total = len(modes)
        axes[row_idx, 0].set_ylabel(shape_name, fontsize=9, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure4_scale_scaling.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.out_dir / 'figure4_scale_scaling.png'}")
    return make_table([{"figure": 4, "status": "ok"}])


def _imshow(ax, img, title, cmap, vmin, vmax, extent=1.0):
    R = float(extent)
    im = ax.imshow(img, extent=(-R, R, -R, R), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(out_dir=out_dir, seed=args.seed))


if __name__ == "__main__":
    main()
