#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3: Frequency / resolution effects with close three blocks.

Layout: 3 rows × 2 cols = 6 reconstructions (no truth column).
  Row 1: k = 4, 6
  Row 2: k = 8, 10
  Row 3: k = 15, 20

Data: analytical block Born far-field, finite-direction mock measurement mode,
far-field noise=0.2.  All frequencies use the same incident/observation
direction set.  Six nearby configurations are selected per target node with a
full-rank polarimetric constraint.
Truncation: epsilon filtering followed by the theoretical, discrete, and hard
dimension caps, without splitting complete ``(ell,n)`` multiplets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig,
    ball_quadrature_nodes,
    collect_alpha_pairs_cached,
    collect_reconstruction_diagnostics,
    generate_polarimetric_data_nodes,
    make_table,
    modal_matrix,
    plot_diagnostic_curves,
    quadrature_modal_coefficients,
    reference_tensor,
    save_diagnostics_npz,
    select_complete_gpswf_multiplets,
    solve_ball_gpswf,
    tensor_coefficients_from_matrix,
    write_diagnostics_csv,
)
from common.phantom import Mode, close_three_block_phantom, truth_image_2d
from forward.datasets import (
    analytic_block_born_farfield_dataset,
    farfield_dataset_to_qhat,
    polarimetric_diagnostic_summary,
)


def _row_params(k: float) -> dict[str, float | int]:
    """Empirical 3D GPSWF discretisation parameters for each frequency."""
    C = 2.0 * k
    parameters = {
        4: {"ell_max": 4, "n_modes": 2, "K": 20, "n_radial": 6, "n_angular": 74},
        6: {"ell_max": 5, "n_modes": 3, "K": 26, "n_radial": 8, "n_angular": 110},
        8: {"ell_max": 7, "n_modes": 3, "K": 32, "n_radial": 10, "n_angular": 146},
        10: {"ell_max": 8, "n_modes": 5, "K": 40, "n_radial": 12, "n_angular": 194},
        15: {"ell_max": 12, "n_modes": 6, "K": 64, "n_radial": 20, "n_angular": 302},
        20: {"ell_max": 14, "n_modes": 7, "K": 80, "n_radial": 28, "n_angular": 434},
    }
    key = int(k)
    if float(key) != float(k) or key not in parameters:
        raise ValueError(f"unsupported Experiment 3 wavenumber: {k}")
    return {**parameters[key], "C": C}


def _row_params_quick(k: float) -> dict[str, float | int]:
    """Quick-mode GPSWF parameters (scaled down)."""
    C = 2.0 * k
    if k <= 4:
        return {"ell_max": 2, "n_modes": 2, "K": 10, "n_radial": 3, "n_angular": 26, "C": C}
    elif k <= 6:
        return {"ell_max": 4, "n_modes": 2, "K": 14, "n_radial": 5, "n_angular": 38, "C": C}
    elif k <= 8:
        return {"ell_max": 5, "n_modes": 3, "K": 16, "n_radial": 5, "n_angular": 50, "C": C}
    elif k <= 9:
        return {"ell_max": 6, "n_modes": 3, "K": 18, "n_radial": 5, "n_angular": 74, "C": C}
    elif k <= 10:
        return {"ell_max": 7, "n_modes": 3, "K": 20, "n_radial": 6, "n_angular": 74, "C": C}
    elif k <= 12:
        return {"ell_max": 8, "n_modes": 3, "K": 22, "n_radial": 6, "n_angular": 86, "C": C}
    elif k <= 16:
        return {"ell_max": 10, "n_modes": 4, "K": 28, "n_radial": 8, "n_angular": 110, "C": C}
    elif k <= 20:
        return {"ell_max": 12, "n_modes": 5, "K": 32, "n_radial": 10, "n_angular": 146, "C": C}
    else:  # k <= 24
        return {"ell_max": 14, "n_modes": 5, "K": 36, "n_radial": 10, "n_angular": 170, "C": C}


def _settings(quick: bool):
    """Return (grid_size, quad_order, r_eval)."""
    if quick:
        return 51, 100, 80
    return 161, 160, 120


def run_experiment(config: ExperimentConfig) -> Any:
    (grid_size, quad_order, r_eval_count) = _settings(config.quick)
    k_pairs = [(4, 6), (8, 10), (15, 20)]

    kind = "full"; component_index = 0
    epsilon = 0.2
    polarimetric_J = 6
    noise_level = 0.2
    requested_measure_dirs = 110 if config.quick else 974

    rng = np.random.default_rng(config.seed + 300)

    # Truth image (same phantom for all k, computed per k for consistency)
    blocks = close_three_block_phantom()
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows = len(k_pairs); n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_panels: list[tuple[np.ndarray, str]] = []

    for row_idx, (k1, k2) in enumerate(k_pairs):
        for col_idx, k in enumerate([k1, k2]):
            rp = _row_params_quick(k) if config.quick else _row_params(k)
            C = float(rp["C"])
            ell_max = int(rp["ell_max"])
            n_modes_per_ell = int(rp["n_modes"])
            K_val = int(rp["K"])
            n_radial = int(rp["n_radial"])
            n_angular = int(rp["n_angular"])
            # Target quadrature
            target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)

            # Analytical Born data at actual finite-direction Fourier nodes.
            _, farfield_inc, farfield_obs, mock_distances, data_info = (
                generate_polarimetric_data_nodes(
                    -target_nodes,
                    requested_measure_dirs,
                    data_mode=config.data_mode,
                    polarimetric_J=polarimetric_J,
                    tensor_kind=kind,
                )
            )
            ds = analytic_block_born_farfield_dataset(
                blocks,
                target_nodes,
                kind=kind,
                k=k,
                incident_dirs=farfield_inc,
                obs_dirs=farfield_obs,
            )
            rec_c = farfield_dataset_to_qhat(
                ds, kind=kind, noise_level=noise_level, rng=rng
            )
            comp_data = rec_c[:, component_index]
            polarimetric_diagnostics = polarimetric_diagnostic_summary(ds)

            # GPSWF basis + truncation
            alpha_df = collect_alpha_pairs_cached(
                C, K_val, ell_max, n_modes_per_ell,
                quad_order=quad_order, r_eval_count=r_eval_count,
                cache_dir=config.out_dir / "alpha_cache",
            )
            alpha_lookup = {
                (int(r["ell"]), int(r["n"])): complex(float(r["alpha_real"]), float(r["alpha_imag"]))
                for _, r in alpha_df.iterrows()
            }
            modes: list[Mode] = []
            chi_lookup: dict[tuple[int, int], float] = {}
            for ell in range(ell_max + 1):
                chi, beta = solve_ball_gpswf(
                    C, ell, K_val, n_modes=n_modes_per_ell
                )
                for n in range(beta.shape[1]):
                    a = alpha_lookup[(ell, n)]
                    chi_lookup[(ell, n)] = float(chi[n])
                    for m in range(-ell, ell + 1):
                        modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))

            alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
            epsilon_eligible = alpha_abs > epsilon * float(np.max(alpha_abs))
            N_cap_theory = int(C * C / 2)
            N_cap_discrete = target_nodes.shape[0] // 6
            N_cap_hard = 512
            N_cap = min(N_cap_theory, N_cap_discrete, N_cap_hard)
            retained, truncation_info = select_complete_gpswf_multiplets(
                modes,
                chi_lookup,
                N_cap,
                eligible=epsilon_eligible,
            )

            # Reconstruction
            target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
            xs = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(xs, xs)
            grid_points = np.column_stack([
                X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)
            ])
            image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

            truth, _, disk_mask = truth_image_2d(grid_size, blocks, coeff0[component_index])
            vmin = float(np.nanmin(np.real(truth)))
            vmax = float(np.nanmax(np.real(truth)))

            coeffs = quadrature_modal_coefficients(
                comp_data, target_basis, target_weights, modes, retained)
            rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
            rec[~disk_mask] = 0.0

            diagnostic_rows.append(collect_reconstruction_diagnostics(
                case={
                    "experiment": 3,
                    "case_id": f"k{k:g}",
                    "row": int(row_idx),
                    "column": int(col_idx),
                    "k": float(k),
                    "C": float(C),
                    "K": int(K_val),
                    "ell_max": int(ell_max),
                    "n_modes_per_ell": int(n_modes_per_ell),
                    "epsilon": float(epsilon),
                    "N_cap": int(N_cap),
                    "N_cap_theory": int(N_cap_theory),
                    "N_cap_discrete": int(N_cap_discrete),
                    "N_cap_hard": int(N_cap_hard),
                    "retained_N": int(np.sum(retained)),
                    "retained_multiplets": int(truncation_info["retained_multiplets"]),
                    "partial_multiplets": int(truncation_info["partial_multiplets"]),
                    "noise_level": float(noise_level),
                    "n_radial": int(n_radial),
                    "n_angular_requested": int(n_angular),
                    "n_geometries": int(polarimetric_J),
                    "requested_measure_dirs": int(requested_measure_dirs),
                    "actual_measure_dirs": int(data_info["n_measure_dirs"]),
                    "candidate_count": int(data_info["candidate_count"]),
                    "data_mode": config.data_mode,
                    "data_source": "analytic_block_born",
                    "shape": "close_three_blocks_gap_0.20",
                    "C_mock_distance_mean": float(C * np.mean(mock_distances)),
                    "C_mock_distance_p95": float(C * np.percentile(mock_distances, 95)),
                    "C_mock_distance_max": float(C * np.max(mock_distances)),
                    **polarimetric_diagnostics,
                },
                modes=modes,
                retained=retained,
                target_nodes=target_nodes,
                p_nodes=ds.p_nodes,
                mock_distances=mock_distances,
                basis_matrix=target_basis,
                target_weights=target_weights,
                component_data=comp_data,
                coeffs=coeffs,
                image=rec,
                truth=truth,
                disk_mask=disk_mask,
            ))

            title = f"k={k:g}, N={int(np.sum(retained))}"
            real_rec = np.real(rec)
            reconstruction_panels.append((real_rec, title))
            _imshow(axes[row_idx, col_idx], real_rec, title, vmin, vmax)

    fig.supylabel(f"noise = {noise_level:g}", fontsize=10)
    fig.savefig(config.out_dir / "exp3_frequency.png", dpi=200)
    plt.close(fig)

    adaptive_fig, adaptive_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.1 * n_rows),
        constrained_layout=True,
    )
    adaptive_axes = np.asarray(adaptive_axes).reshape(n_rows, n_cols)
    for flat_idx, (image, title) in enumerate(reconstruction_panels):
        ax = adaptive_axes[flat_idx // n_cols, flat_idx % n_cols]
        valid_values = image[disk_mask]
        panel_vmin = float(np.nanmin(valid_values))
        panel_vmax = float(np.nanmax(valid_values))
        im = _imshow(ax, image, title, panel_vmin, panel_vmax)
        adaptive_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    adaptive_fig.supylabel(f"noise = {noise_level:g}", fontsize=10)
    adaptive_fig.savefig(
        config.out_dir / "exp3_frequency_individual_scale.png",
        dpi=200,
    )
    plt.close(adaptive_fig)

    write_diagnostics_csv(diagnostic_rows, config.out_dir / "exp3_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "exp3_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "exp3_diagnostic_curves.png",
        title="Experiment 3 diagnostics",
    )
    print(f"Saved {config.out_dir / 'exp3_frequency.png'}")
    return make_table([{"experiment": 3, "status": "ok"}])


def _imshow(ax, img, title, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower",
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="bicubic")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: frequency effects (close three blocks)")
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-mode", choices=["mock", "ideal"], default="mock")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(
        out_dir=out_dir,
        seed=args.seed,
        quick=args.quick,
        data_mode=args.data_mode,
    ))


if __name__ == "__main__":
    main()
