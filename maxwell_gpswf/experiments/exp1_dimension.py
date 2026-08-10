#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1: Truncation dimension effects with a single cube block.

Layout: 3 rows x 3 cols = 9 reconstructions (no truth column).
  Row 1: N = 1, 5, 21
  Row 2: N = 35, 57, 71
  Row 3: N = 135, 237, 496

Data: VIE Born far-field, finite-direction mock measurement mode, k=15,
far-field noise=0.2.
Truncation: sort complete (ell, n) multiplets by |alpha| and use N as an
upper bound without splitting the m degeneracy.
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
from common.phantom import Mode, cube_phantom, truth_image_2d
from forward.datasets import (
    discrete_vie_born_farfield_dataset,
    farfield_dataset_to_qhat,
    polarimetric_diagnostic_summary,
)


def _settings(quick: bool):
    """Return (n_measure, n_radial, n_angular, grid_size, n_per_axis, N_values, quad_order, r_eval)."""
    if quick:
        # Smaller N set, lighter quadrature
        return 110, 10, 170, 51, 7, [1, 5, 21, 35, 57, 71, 135, 135, 135], 100, 80
    return 974, 12, 230, 161, 19, [1, 5, 21, 35, 57, 71, 135, 237, 496], 160, 120


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, n_angular, grid_size,
     n_per_axis, N_values, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k
    kind = "full"; component_index = 0
    polarimetric_J = 6
    noise_level = 0.2
    cube_half_side = 0.4

    # GPSWF params for k=15 (figure1/figure3 pattern)
    ell_max = 12
    n_modes_per_ell = 7
    K_val = 48

    rng = np.random.default_rng(config.seed + 100)

    # -- Target quadrature nodes --
    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)

    # -- VIE Born data with mock-matched direction pairs (figure3 pattern) --
    vie_physical, vie_inc, vie_obs, vie_dist, data_info = generate_polarimetric_data_nodes(
        -target_nodes,
        requested_measure_dirs,
        data_mode=config.data_mode,
        polarimetric_J=polarimetric_J,
        tensor_kind=kind,
    )
    vie_nodes = -vie_physical

    ds = discrete_vie_born_farfield_dataset(
        "cube", vie_nodes, kind=kind, k=k, R=1.0,
        n_per_axis=n_per_axis, n_geometries=polarimetric_J,
        incident_dirs=vie_inc, obs_dirs=vie_obs,
        cube_half_side=cube_half_side)
    rec_c = farfield_dataset_to_qhat(
        ds, kind=kind, noise_level=noise_level, rng=rng
    )
    comp_data = rec_c[:, component_index]
    polarimetric_diagnostics = polarimetric_diagnostic_summary(ds)

    # -- Truth image (for vmin/vmax) --
    blocks = cube_phantom(center=(0.0, 0.0, 0.0), half_side=cube_half_side,
                          amplitude=1.0 + 0.0j)
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
    chi_lookup: dict[tuple[int, int], float] = {}
    for ell in range(ell_max + 1):
        chi, beta = solve_ball_gpswf(C, ell, K_val, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            a = alpha_lookup[(ell, n)]
            chi_lookup[(ell, n)] = float(chi[n])
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))

    target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
    image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

    # -- Plot: 3 rows x 3 cols --
    N_rows = 3; N_cols = 3
    fig, axes = plt.subplots(N_rows, N_cols,
                             figsize=(3.1 * N_cols, 3.1 * N_rows),
                             constrained_layout=True)
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_panels: list[tuple[np.ndarray, str]] = []

    for flat_idx, N in enumerate(N_values):
        row_idx = flat_idx // N_cols
        col_idx = flat_idx % N_cols

        requested_dim = min(N, len(modes))
        retained, truncation_info = select_complete_gpswf_multiplets(
            modes,
            chi_lookup,
            requested_dim,
        )
        dim = int(np.sum(retained))

        coeffs = quadrature_modal_coefficients(
            comp_data, target_basis, target_weights, modes, retained)
        rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
        rec[~disk_mask] = 0.0

        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                "experiment": 1,
                "case_id": f"N{dim}",
                "row": int(row_idx),
                "column": int(col_idx),
                "k": float(k),
                "C": float(C),
                "K": int(K_val),
                "ell_max": int(ell_max),
                "n_modes_per_ell": int(n_modes_per_ell),
                "requested_N": int(N),
                "retained_N": int(dim),
                "retained_multiplets": int(truncation_info["retained_multiplets"]),
                "partial_multiplets": int(truncation_info["partial_multiplets"]),
                "alpha_plateau_rtol": float(truncation_info["alpha_plateau_rtol"]),
                "noise_level": float(noise_level),
                "n_radial": int(n_radial),
                "n_angular_requested": int(n_angular),
                "n_per_axis": int(n_per_axis),
                "n_geometries": int(polarimetric_J),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                "data_mode": config.data_mode,
                "data_source": "vie_born",
                "shape": "cube",
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

        title = f"N={dim}" if dim == N else f"N<={N}, N_eff={dim}"
        real_rec = np.real(rec)
        reconstruction_panels.append((real_rec, title))
        _imshow(axes[row_idx, col_idx], real_rec, title, vmin, vmax)

    fig.supylabel(
        f"k = {k:g}, noise = {noise_level:g}",
        fontsize=10,
    )

    fig.savefig(config.out_dir / "exp1_dimension.png", dpi=200)
    plt.close(fig)

    adaptive_fig, adaptive_axes = plt.subplots(
        N_rows,
        N_cols,
        figsize=(3.8 * N_cols, 3.1 * N_rows),
        constrained_layout=True,
    )
    for flat_idx, (image, title) in enumerate(reconstruction_panels):
        ax = adaptive_axes[flat_idx // N_cols, flat_idx % N_cols]
        valid_values = image[disk_mask]
        panel_vmin = float(np.nanmin(valid_values))
        panel_vmax = float(np.nanmax(valid_values))
        im = _imshow(ax, image, title, panel_vmin, panel_vmax)
        adaptive_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    adaptive_fig.supylabel(
        f"k = {k:g}, noise = {noise_level:g}",
        fontsize=10,
    )
    adaptive_fig.savefig(
        config.out_dir / "exp1_dimension_individual_scale.png",
        dpi=200,
    )
    plt.close(adaptive_fig)

    write_diagnostics_csv(diagnostic_rows, config.out_dir / "exp1_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "exp1_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "exp1_diagnostic_curves.png",
        title="Experiment 1 diagnostics",
    )
    print(f"Saved {config.out_dir / 'exp1_dimension.png'}")
    return make_table([{"experiment": 1, "status": "ok"}])


def _imshow(ax, img, title, vmin, vmax):
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower",
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="bicubic")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: dimension effects (single cube)")
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
