#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2: Noise effects with three-block phantom.

Layout: 1 row x 3 cols = 3 reconstructions (no truth column).
  Noise = 0, 0.2, 0.4

Data: VIE Born far-field, mock measurement mode, k=15.
Truncation: three-layer (GPSWF params -> epsilon=0.2 -> N_cap=C^2/2).
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
    solve_ball_gpswf,
    tensor_coefficients_from_matrix,
    write_diagnostics_csv,
)
from common.phantom import Mode, three_block_phantom, truth_image_2d
from forward.datasets import (
    discrete_vie_born_farfield_dataset,
    farfield_dataset_to_qhat,
    polarimetric_diagnostic_summary,
)


def _settings(quick: bool):
    """Return (n_measure, n_radial, n_angular, grid_size, n_per_axis, quad_order, r_eval)."""
    if quick:
        return 110, 10, 170, 51, 7, 100, 80
    return 974, 12, 230, 161, 19, 160, 120


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, n_angular, grid_size,
     n_per_axis, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k
    kind = "full"; component_index = 0
    epsilon = 0.2
    noise_levels = [0.0, 0.2, 0.4]
    polarimetric_J = 6

    ell_max = 12; n_modes_per_ell = 7; K_val = 48

    rng = np.random.default_rng(config.seed + 200)

    # -- Target quadrature nodes --
    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)

    # -- VIE Born data with mock-matched direction pairs (figure3 pattern) --
    vie_physical, vie_inc, vie_obs, vie_dist, data_info = generate_polarimetric_data_nodes(
        -target_nodes,
        requested_measure_dirs,
        polarimetric_J=polarimetric_J,
        tensor_kind=kind,
    )
    vie_nodes = -vie_physical

    ds = discrete_vie_born_farfield_dataset(
        "three_blocks", vie_nodes, kind=kind, k=k, R=1.0,
        n_per_axis=n_per_axis, n_geometries=polarimetric_J,
        incident_dirs=vie_inc, obs_dirs=vie_obs)

    # -- Truth image (for vmin/vmax) --
    blocks = three_block_phantom("born")
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)
    truth, grid_points, disk_mask = truth_image_2d(grid_size, blocks, coeff0[component_index])
    vmin = float(np.nanmin(np.real(truth)))
    vmax = float(np.nanmax(np.real(truth)))

    # -- Build GPSWF basis --
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
    for ell in range(ell_max + 1):
        _, beta = solve_ball_gpswf(C, ell, K_val, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            a = alpha_lookup[(ell, n)]
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))

    # Three-layer truncation
    alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
    retained = alpha_abs > epsilon * float(np.max(alpha_abs))
    N_cap = int(C * C / 2)
    if np.sum(retained) > N_cap:
        order = np.argsort(-alpha_abs)
        keep = order[:N_cap]
        retained = np.zeros(len(modes), dtype=bool)
        retained[keep] = True

    target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
    image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

    # -- Plot: 1 row x 3 cols --
    N_rows = 1; N_cols = 3
    fig, axes = plt.subplots(N_rows, N_cols,
                             figsize=(3.1 * N_cols, 3.1 * N_rows),
                             constrained_layout=True)
    if N_rows == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_panels: list[tuple[np.ndarray, str]] = []

    for col_idx, noise_level in enumerate(noise_levels):
        rec_c = farfield_dataset_to_qhat(
            ds, kind=kind, noise_level=noise_level, rng=rng)
        comp_data = rec_c[:, component_index]
        polarimetric_diagnostics = polarimetric_diagnostic_summary(ds)

        coeffs = quadrature_modal_coefficients(
            comp_data, target_basis, target_weights, modes, retained)
        rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
        rec[~disk_mask] = 0.0

        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                "experiment": 2,
                "case_id": f"noise{noise_level:g}",
                "row": 0,
                "column": int(col_idx),
                "k": float(k),
                "C": float(C),
                "K": int(K_val),
                "ell_max": int(ell_max),
                "n_modes_per_ell": int(n_modes_per_ell),
                "epsilon": float(epsilon),
                "N_cap": int(N_cap),
                "retained_N": int(np.sum(retained)),
                "noise_level": float(noise_level),
                "n_radial": int(n_radial),
                "n_angular_requested": int(n_angular),
                "n_per_axis": int(n_per_axis),
                "n_geometries": int(polarimetric_J),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                "data_mode": "mock",
                "data_source": "vie_born",
                "shape": "three_blocks",
                **polarimetric_diagnostics,
            },
            modes=modes,
            retained=retained,
            target_nodes=target_nodes,
            p_nodes=vie_nodes,
            mock_distances=vie_dist,
            basis_matrix=target_basis,
            target_weights=target_weights,
            component_data=comp_data,
            coeffs=coeffs,
            image=rec,
            truth=truth,
            disk_mask=disk_mask,
        ))

        title = f"noise={noise_level:g}"
        real_rec = np.real(rec)
        reconstruction_panels.append((real_rec, title))
        _imshow(axes[0, col_idx], real_rec, title, vmin, vmax)

    retained_N = int(np.sum(retained))
    fig.supylabel(f"k = {k:g}, N = {retained_N}", fontsize=10)

    fig.savefig(config.out_dir / "exp2_noise.png", dpi=200)
    plt.close(fig)

    adaptive_fig, adaptive_axes = plt.subplots(
        N_rows,
        N_cols,
        figsize=(3.8 * N_cols, 3.1 * N_rows),
        constrained_layout=True,
    )
    adaptive_axes = np.atleast_1d(adaptive_axes)
    for col_idx, (image, title) in enumerate(reconstruction_panels):
        valid_values = image[disk_mask]
        panel_vmin = float(np.nanmin(valid_values))
        panel_vmax = float(np.nanmax(valid_values))
        im = _imshow(
            adaptive_axes[col_idx],
            image,
            title,
            panel_vmin,
            panel_vmax,
        )
        adaptive_fig.colorbar(
            im,
            ax=adaptive_axes[col_idx],
            fraction=0.046,
            pad=0.03,
        )
    adaptive_fig.supylabel(f"k = {k:g}, N = {retained_N}", fontsize=10)
    adaptive_fig.savefig(
        config.out_dir / "exp2_noise_individual_scale.png",
        dpi=200,
    )
    plt.close(adaptive_fig)

    write_diagnostics_csv(diagnostic_rows, config.out_dir / "exp2_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "exp2_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "exp2_diagnostic_curves.png",
        title="Experiment 2 diagnostics",
    )
    print(f"Saved {config.out_dir / 'exp2_noise.png'}")
    return make_table([{"experiment": 2, "status": "ok"}])


def _imshow(ax, img, title, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower",
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="bicubic")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: noise effects (three blocks)")
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(
        out_dir=out_dir,
        seed=args.seed,
        quick=args.quick,
    ))


if __name__ == "__main__":
    main()
