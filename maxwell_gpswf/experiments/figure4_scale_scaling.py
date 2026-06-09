#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4: Support-radius scaling — prior support radius R varies.

Layout: 5 cols (truth + R=1.0/1.5/2.0/3.0) × 5 rows (shapes from fig3).
Data: analytical Born for a fixed physical scatterer Q(x).

For supp(Q) ⊂ B(0,R), set x = R y and C = 2 k R.  The unit-ball GPSWF
inversion reconstructs f_R(y) = R^3 Q(R y); plotting physical Q(x) therefore
evaluates at y=x/R and divides the result by R^3.
"""
from __future__ import annotations

import argparse, math; from pathlib import Path; from typing import Any
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np

from common import (
    collect_reconstruction_diagnostics,
    ExperimentConfig, ball_quadrature_nodes, collect_alpha_pairs_cached,
    generate_data_nodes, make_table, modal_matrix,
    plot_diagnostic_curves,
    quadrature_modal_coefficients, recover_polarimetric_coefficients,
    reference_tensor, solve_ball_gpswf,
    save_diagnostics_npz,
    tensor_coefficients_from_matrix,
    write_diagnostics_csv,
)
from common.phantom import Mode, _shape_truth_and_fourier


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
    data_C = 2.0 * k
    quad_order = 160; r_eval_count = 120
    data_mode = getattr(config, "data_mode", "mock")

    rng = np.random.default_rng(config.seed + 400)
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows = len(SHAPES); n_cols = 1 + len(R_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    diagnostic_rows: list[dict[str, Any]] = []

    for row_idx, shape_name in enumerate(SHAPES):
        print(f"  Processing {shape_name}...")
        truth_ref, _, _, _ = _shape_truth_and_fourier(
            shape_name, np.zeros((1, 3), dtype=float), grid_size, data_C)
        vmin_ref = float(np.nanmin(np.real(truth_ref)))
        vmax_ref = float(np.nanmax(np.real(truth_ref)))

        for col_idx, R in enumerate([None] + R_VALUES):
            if R is None:
                _imshow(axes[row_idx, 0], np.real(truth_ref),
                        "truth" if row_idx == 0 else "", "viridis",
                        vmin_ref, vmax_ref)
                continue

            rp = _row_params(R); C = rp["C"]
            ell_max = rp["ell_max"]; n_modes = rp["n_modes"]; K_val = rp["K"]
            n_radial = rp["n_radial"]; n_angular = rp["n_angular"]

            # Quadrature
            target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)
            p_nodes, _, _, mock_distances, data_info = generate_data_nodes(
                target_nodes, requested_measure_dirs, data_mode=data_mode, branch_count=1)

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
            physical_points = np.column_stack([
                X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)
            ])
            normalized_points = physical_points / float(R)
            image_matrix = modal_matrix(normalized_points, modes, fourier_side=False)

            # Analytical Born data of the fixed physical scatterer Q(x).
            # Under x=R y, this equals the Fourier data of f_R(y)=R^3 Q(Ry)
            # at bandwidth C=2kR.
            _, _, _, fourier_data = _shape_truth_and_fourier(shape_name, p_nodes, grid_size, data_C)
            rec_c, _, _ = recover_polarimetric_coefficients(
                p_nodes, fourier_data[:, None] * coeff0[None, :], kind, 0.0, rng)
            comp_data = rec_c[:, component_index]
            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size) / (float(R) ** 3)
            support_mask = np.sum(physical_points * physical_points, axis=1).reshape(grid_size, grid_size) <= float(R) ** 2
            rec[~support_mask] = 0.0
            diagnostic_rows.append(collect_reconstruction_diagnostics(
                case={
                    "figure": 4,
                    "case_id": f"{shape_name}_R{R:g}",
                    "row": int(row_idx),
                    "column": int(col_idx),
                    "shape_name": shape_name,
                    "R": float(R),
                    "k": float(k),
                    "C": float(C),
                    "K": int(K_val),
                    "ell_max": int(ell_max),
                    "n_modes_per_ell": int(n_modes),
                    "epsilon": float(epsilon),
                    "N_cap": int(N_cap),
                    "n_radial": int(n_radial),
                    "n_angular_requested": int(n_angular),
                    "data_mode": data_mode,
                    "n_measure_dirs": int(data_info["n_measure_dirs"]),
                    "measure_rule": data_info["measure_rule"],
                    "noise_level": 0.0,
                    "contrast_scale": math.nan,
                    "data_source": "analytical_born_fixed_physical_scatterer",
                },
                modes=modes,
                retained=retained,
                target_nodes=target_nodes,
                p_nodes=p_nodes,
                mock_distances=mock_distances,
                basis_matrix=target_basis,
                target_weights=target_weights,
                component_data=comp_data,
                coeffs=coeffs,
                image=rec,
                truth=truth_ref,
                disk_mask=np.ones_like(truth_ref, dtype=bool),
            ))

            label = f"R={R}" if row_idx == 0 else ""
            _imshow(axes[row_idx, col_idx], np.real(rec), label, "viridis",
                    vmin_ref, vmax_ref)

        n_total = len(modes)
        axes[row_idx, 0].set_ylabel(shape_name, fontsize=9, rotation=90, labelpad=12)

    fig.savefig(config.out_dir / "figure4_scale_scaling.png", dpi=200)
    plt.close(fig)
    write_diagnostics_csv(diagnostic_rows, config.out_dir / "figure4_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "figure4_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "figure4_diagnostic_curves.png",
        title="Figure 4 diagnostics",
    )
    print(f"Saved {config.out_dir / 'figure4_scale_scaling.png'}")
    return make_table([{"figure": 4, "status": "ok"}])


def _imshow(ax, img, title, cmap, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--data-mode", choices=["mock", "ideal"], default="mock",
                   help="mock: nearest measured Fourier node; ideal: exact admissible direction pairs")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(out_dir=out_dir, seed=args.seed, data_mode=args.data_mode))


if __name__ == "__main__":
    main()
