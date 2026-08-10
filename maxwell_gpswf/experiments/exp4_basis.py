#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 4: Basis comparison - GPSWF, Cube Fourier, Ball Bessel, DSM.

Layout: 3 rows x 4 cols = 12 reconstructions (no truth column).
  Rows: k = 8, 12, 15
  Cols: GPSWF | Fourier | Bessel | DSM

Data: VIE Born far-field, finite-direction mock measurement mode,
far-field noise=0.2.  Six nearby direction configurations are selected per
target node with a full-rank polarimetric constraint.
All methods use the same VIE Born recovered component data for fair comparison.
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
    direct_sampling_component_indicator,
    generate_polarimetric_data_nodes,
    make_table,
    modal_matrix,
    plot_diagnostic_curves,
    quadrature_modal_coefficients,
    reconstruct_ball_bessel_from_data,
    reconstruct_fourier_cube_from_data,
    reference_tensor,
    save_diagnostics_npz,
    select_complete_gpswf_multiplets,
    solve_ball_gpswf,
    tensor_coefficients_from_matrix,
    write_diagnostics_csv,
)
from common.phantom import Mode, three_block_phantom, truth_image_2d
from common.utils import vector_norm
from forward.datasets import (
    discrete_vie_born_farfield_dataset,
    farfield_dataset_to_qhat,
    polarimetric_diagnostic_summary,
)


def _row_params(k: float) -> dict[str, float | int]:
    """GPSWF parameters linked to k (extended for k=12,16,20,24)."""
    C = 2.0 * k
    if k <= 4:
        return {"ell_max": 4, "n_modes": 2, "K": 16, "n_radial": 5, "n_angular": 50, "C": C}
    elif k <= 6:
        return {"ell_max": 5, "n_modes": 3, "K": 22, "n_radial": 6, "n_angular": 74, "C": C}
    elif k <= 7:
        return {"ell_max": 6, "n_modes": 3, "K": 24, "n_radial": 7, "n_angular": 74, "C": C}
    elif k <= 8:
        return {"ell_max": 7, "n_modes": 3, "K": 28, "n_radial": 8, "n_angular": 86, "C": C}
    elif k <= 9:
        return {"ell_max": 7, "n_modes": 4, "K": 32, "n_radial": 8, "n_angular": 110, "C": C}
    elif k <= 10:
        return {"ell_max": 8, "n_modes": 5, "K": 36, "n_radial": 10, "n_angular": 110, "C": C}
    elif k <= 12:
        return {"ell_max": 9, "n_modes": 5, "K": 40, "n_radial": 10, "n_angular": 170, "C": C}
    elif k <= 16:
        return {"ell_max": 12, "n_modes": 6, "K": 48, "n_radial": 12, "n_angular": 230, "C": C}
    elif k <= 20:
        return {"ell_max": 14, "n_modes": 7, "K": 54, "n_radial": 14, "n_angular": 302, "C": C}
    else:  # k <= 24
        return {"ell_max": 16, "n_modes": 7, "K": 60, "n_radial": 14, "n_angular": 302, "C": C}


def _row_params_quick(k: float) -> dict[str, float | int]:
    """Quick-mode GPSWF parameters."""
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


def _n_per_axis_for_k(k: float, quick: bool) -> int:
    if quick:
        return 7
    return max(11, min(19, int(k * 1.2)))


def _settings(quick: bool):
    """Return (grid_size, quad_order, r_eval)."""
    if quick:
        return 51, 100, 80
    return 161, 160, 120


def _baseline_diagnostics(
    *,
    case: dict[str, Any],
    image: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(case)
    abs_img = np.abs(image)
    target_mask = valid_mask & (np.abs(truth) > 1e-12)
    background_mask = valid_mask & ~target_mask

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_img[mask])) if np.any(mask) else float("nan")

    def p95(mask: np.ndarray) -> float:
        return float(np.percentile(abs_img[mask], 95)) if np.any(mask) else float("nan")

    row.update({
        "retained_modes": int(row.get("basis_modes", row.get("retained_N", 0))),
        "total_modes": int(row.get("basis_modes", row.get("total_modes", 0))),
        "gram_offdiag_ratio": float("nan"),
        "gram_cond": float("nan"),
        "data_norm": float(row.get("data_norm", float("nan"))),
        "data_max_abs": float(row.get("data_max_abs", float("nan"))),
        "image_min": float(np.min(np.real(image))),
        "image_max": float(np.max(np.real(image))),
        "image_max_abs": float(np.max(abs_img)),
        "image_l2": vector_norm(image),
        "background_mean_abs": mean(background_mask),
        "background_p95_abs": p95(background_mask),
        "target_mean_abs": mean(target_mask),
        "target_p95_abs": p95(target_mask),
    })
    return row


def run_experiment(config: ExperimentConfig) -> Any:
    (grid_size, quad_order, r_eval_count) = _settings(config.quick)
    k_values = [8, 12, 15]

    kind = "full"; component_index = 0
    epsilon = 0.2
    polarimetric_J = 6
    noise_level = 0.2
    requested_measure_dirs = 110 if config.quick else 974
    half_side = 1.0
    bandwidth_factor = 2.0
    basis_lstsq_rcond = 1e-8
    fourier_mode_fraction = 1.2
    bessel_mode_fraction = 1.2
    basis_mode_min = 12

    rng = np.random.default_rng(config.seed + 400)

    blocks = three_block_phantom("born")
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)

    n_rows = len(k_values); n_cols = 4  # GPSWF, Fourier, Bessel, DSM
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.1 * n_cols, 3.1 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_panels: list[tuple[np.ndarray, str, np.ndarray]] = []

    for row_idx, k in enumerate(k_values):
        rp = _row_params_quick(k) if config.quick else _row_params(k)
        C = float(rp["C"])
        ell_max = int(rp["ell_max"])
        n_modes_per_ell = int(rp["n_modes"])
        K_val = int(rp["K"])
        n_radial = int(rp["n_radial"])
        n_angular = int(rp["n_angular"])
        n_per_axis = _n_per_axis_for_k(k, config.quick)

        # Target quadrature
        target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)

        # VIE Born data on finite measured directions matched to -p.
        vie_physical, vie_inc, vie_obs, vie_dist, data_info = (
            generate_polarimetric_data_nodes(
                -target_nodes,
                requested_measure_dirs,
                data_mode=config.data_mode,
                polarimetric_J=polarimetric_J,
                tensor_kind=kind,
            )
        )
        vie_nodes = -vie_physical
        ds = discrete_vie_born_farfield_dataset(
            "three_blocks", vie_nodes, kind=kind, k=k, R=1.0,
            n_per_axis=n_per_axis, n_geometries=polarimetric_J,
            incident_dirs=vie_inc, obs_dirs=vie_obs,
        )
        rec_c = farfield_dataset_to_qhat(
            ds, kind=kind, noise_level=noise_level, rng=rng
        )
        comp_data = rec_c[:, component_index]
        polarimetric_diagnostics = polarimetric_diagnostic_summary(ds)

        # Truth (for vmin/vmax)
        truth, _, disk_mask = truth_image_2d(grid_size, blocks, coeff0[component_index])

        # Grid
        xs = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(xs, xs)
        grid_points = np.column_stack([
            X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)
        ])
        vmin = float(np.nanmin(np.real(truth)))
        vmax = float(np.nanmax(np.real(truth)))
        square_mask = np.ones_like(disk_mask, dtype=bool)

        # -- GPSWF basis --
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
        gpswf_retained_count = int(np.sum(retained))
        fourier_mode_cap = min(
            N_cap_hard,
            max(
                int(basis_mode_min),
                int(fourier_mode_fraction * gpswf_retained_count),
            ),
        )
        bessel_mode_cap = min(
            N_cap_hard,
            max(
                int(basis_mode_min),
                int(bessel_mode_fraction * gpswf_retained_count),
            ),
        )

        target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
        image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

        # --- GPSWF reconstruction ---
        gpswf_coeffs = quadrature_modal_coefficients(
            comp_data, target_basis, target_weights, modes, retained)
        gpswf_rec = (image_matrix @ gpswf_coeffs).reshape(grid_size, grid_size)
        gpswf_rec[~disk_mask] = 0.0
        gpswf_image = np.real(gpswf_rec)
        gpswf_title = f"GPSWF\nN={gpswf_retained_count}"
        reconstruction_panels.append((gpswf_image, gpswf_title, disk_mask))
        _imshow(axes[row_idx, 0], gpswf_image, gpswf_title, vmin, vmax)
        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                "experiment": 4,
                "case_id": f"k{k:g}_gpswf",
                "method": "gpswf",
                "row": int(row_idx),
                "column": 0,
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
                "retained_N": gpswf_retained_count,
                "retained_multiplets": int(truncation_info["retained_multiplets"]),
                "partial_multiplets": int(truncation_info["partial_multiplets"]),
                "noise_level": float(noise_level),
                "n_radial": int(n_radial),
                "n_angular_requested": int(n_angular),
                "n_per_axis": int(n_per_axis),
                "n_geometries": int(polarimetric_J),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                "data_mode": config.data_mode,
                "data_source": "vie_born",
                "shape": "three_blocks",
                "support": "unit_ball",
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
            coeffs=gpswf_coeffs,
            image=gpswf_rec,
            truth=truth,
            disk_mask=disk_mask,
        ))

        # --- Fourier reconstruction ---
        data_nodes = target_nodes
        fourier_values, fourier_meta = reconstruct_fourier_cube_from_data(
            comp_data, data_nodes, target_weights, grid_points,
            float(k), C, half_side=half_side,
            bandwidth_factor=bandwidth_factor,
            max_modes=fourier_mode_cap, rcond=basis_lstsq_rcond,
        )
        fourier_rec = fourier_values.reshape(grid_size, grid_size)
        fourier_image = np.real(fourier_rec)
        fourier_title = f"Cube Fourier\nN={fourier_meta.get('fourier_modes', '?')}"
        reconstruction_panels.append((fourier_image, fourier_title, square_mask))
        _imshow(axes[row_idx, 1], fourier_image, fourier_title, vmin, vmax)
        diagnostic_rows.append(_baseline_diagnostics(
            case={
                "experiment": 4,
                "case_id": f"k{k:g}_fourier",
                "method": "cube_fourier",
                "row": int(row_idx),
                "column": 1,
                "k": float(k),
                "C": float(C),
                "noise_level": float(noise_level),
                "data_mode": config.data_mode,
                "data_source": "vie_born",
                "shape": "three_blocks",
                "support": "cube_half_side_1",
                "basis_modes": int(fourier_meta.get("fourier_modes", 0)),
                "basis_mode_fraction": float(fourier_mode_fraction),
                "basis_mode_cap": int(fourier_mode_cap),
                "basis_mode_hard_cap": int(N_cap_hard),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                **polarimetric_diagnostics,
                **fourier_meta,
            },
            image=fourier_rec,
            truth=truth,
            valid_mask=square_mask,
        ))

        # --- Bessel reconstruction ---
        bessel_values, bessel_meta = reconstruct_ball_bessel_from_data(
            comp_data, data_nodes, target_weights, grid_points,
            float(k), C, quadrature_nodes=target_nodes,
            quadrature_weights=target_weights,
            bandwidth_factor=bandwidth_factor,
            max_modes=bessel_mode_cap, rcond=basis_lstsq_rcond,
        )
        bessel_rec = bessel_values.reshape(grid_size, grid_size)
        bessel_rec[~disk_mask] = 0.0
        bessel_image = np.real(bessel_rec)
        bessel_title = f"Ball Bessel\nN={bessel_meta.get('bessel_modes', '?')}"
        reconstruction_panels.append((bessel_image, bessel_title, disk_mask))
        _imshow(axes[row_idx, 2], bessel_image, bessel_title, vmin, vmax)
        diagnostic_rows.append(_baseline_diagnostics(
            case={
                "experiment": 4,
                "case_id": f"k{k:g}_bessel",
                "method": "ball_bessel",
                "row": int(row_idx),
                "column": 2,
                "k": float(k),
                "C": float(C),
                "noise_level": float(noise_level),
                "data_mode": config.data_mode,
                "data_source": "vie_born",
                "shape": "three_blocks",
                "support": "unit_ball",
                "basis_modes": int(bessel_meta.get("bessel_modes", 0)),
                "basis_mode_fraction": float(bessel_mode_fraction),
                "basis_mode_cap": int(bessel_mode_cap),
                "basis_mode_hard_cap": int(N_cap_hard),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                **polarimetric_diagnostics,
                **bessel_meta,
            },
            image=bessel_rec,
            truth=truth,
            valid_mask=disk_mask,
        ))

        # --- DSM reconstruction ---
        dsm_values, dsm_meta = direct_sampling_component_indicator(
            comp_data, data_nodes, target_weights, grid_points, C,
            normalize=True,
        )
        dsm_rec = dsm_values.reshape(grid_size, grid_size)
        dsm_rec[~disk_mask] = 0.0
        truth_scale = max(float(np.nanmax(np.abs(truth))), 1e-14)
        dsm_display = dsm_rec * truth_scale
        dsm_image = np.real(dsm_display)
        dsm_title = f"DSM\nnodes={dsm_meta.get('dsm_nodes', '?')}"
        reconstruction_panels.append((dsm_image, dsm_title, disk_mask))
        _imshow(axes[row_idx, 3], dsm_image, dsm_title, vmin, vmax)
        diagnostic_rows.append(_baseline_diagnostics(
            case={
                "experiment": 4,
                "case_id": f"k{k:g}_dsm",
                "method": "dsm",
                "row": int(row_idx),
                "column": 3,
                "k": float(k),
                "C": float(C),
                "noise_level": float(noise_level),
                "data_mode": config.data_mode,
                "data_source": "vie_born",
                "shape": "three_blocks",
                "support": "unit_ball",
                "basis_modes": int(dsm_meta.get("dsm_nodes", 0)),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                **polarimetric_diagnostics,
                **dsm_meta,
            },
            image=dsm_display,
            truth=truth,
            valid_mask=disk_mask,
        ))

        axes[row_idx, 0].set_ylabel(
            f"k = {k:g}",
            fontsize=9,
            rotation=90,
            labelpad=18,
        )

    fig.suptitle(f"noise = {noise_level:g}", fontsize=10)
    fig.savefig(config.out_dir / "exp4_basis.png", dpi=200)
    plt.close(fig)

    adaptive_fig, adaptive_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.1 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        adaptive_axes = adaptive_axes[None, :]
    for flat_idx, (image, title, valid_mask) in enumerate(reconstruction_panels):
        row_idx, col_idx = divmod(flat_idx, n_cols)
        ax = adaptive_axes[row_idx, col_idx]
        valid_values = image[valid_mask]
        panel_vmin = float(np.nanmin(valid_values))
        panel_vmax = float(np.nanmax(valid_values))
        im = _imshow(ax, image, title, panel_vmin, panel_vmax)
        adaptive_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if col_idx == 0:
            ax.set_ylabel(
                f"k = {k_values[row_idx]:g}",
                fontsize=9,
                rotation=90,
                labelpad=18,
            )
    adaptive_fig.suptitle(f"noise = {noise_level:g}", fontsize=10)
    adaptive_fig.savefig(
        config.out_dir / "exp4_basis_individual_scale.png",
        dpi=200,
    )
    plt.close(adaptive_fig)

    write_diagnostics_csv(diagnostic_rows, config.out_dir / "exp4_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "exp4_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "exp4_diagnostic_curves.png",
        title="Experiment 4 diagnostics",
    )
    print(f"Saved {config.out_dir / 'exp4_basis.png'}")
    return make_table([{"experiment": 4, "status": "ok"}])


def _imshow(ax, img, title, vmin, vmax):
    im = ax.imshow(
        img,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4: basis comparison")
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
