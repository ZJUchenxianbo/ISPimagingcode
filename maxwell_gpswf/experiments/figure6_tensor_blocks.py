#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6: reconstruction of blocks with different tensor contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig,
    ball_quadrature_nodes,
    collect_alpha_pairs_cached,
    direct_sampling_farfield_indicator,
    direct_sampling_tensor_indicator,
    generate_data_nodes,
    make_table,
    modal_matrix,
    polarimetric_farfield_data,
    plot_diagnostic_curves,
    quadrature_modal_coefficients,
    recover_polarimetric_coefficients_from_data,
    reconstruct_ball_bessel_from_data,
    reconstruct_fourier_cube_from_data,
    save_diagnostics_npz,
    solve_ball_gpswf,
    tensor_basis,
    tensor_block_fourier_coefficients,
    tensor_truth_image_2d,
    three_tensor_block_phantom,
    write_diagnostics_csv,
)
from common.phantom import Mode
from common.utils import vector_norm


def _row_params(k: float) -> dict[str, float | int]:
    C = 2.0 * k
    if k <= 4:
        return {"ell_max": 4, "n_modes": 2, "K": 16, "n_radial": 5, "n_angular": 50, "C": C}
    if k <= 6:
        return {"ell_max": 5, "n_modes": 3, "K": 22, "n_radial": 6, "n_angular": 74, "C": C}
    if k <= 7:
        return {"ell_max": 6, "n_modes": 3, "K": 24, "n_radial": 7, "n_angular": 74, "C": C}
    if k <= 8:
        return {"ell_max": 7, "n_modes": 3, "K": 28, "n_radial": 8, "n_angular": 86, "C": C}
    if k <= 9:
        return {"ell_max": 7, "n_modes": 4, "K": 32, "n_radial": 8, "n_angular": 110, "C": C}
    return {"ell_max": 8, "n_modes": 5, "K": 36, "n_radial": 10, "n_angular": 110, "C": C}


def _row_params_quick(k: float) -> dict[str, float | int]:
    C = 2.0 * k
    if k <= 4:
        return {"ell_max": 2, "n_modes": 2, "K": 10, "n_radial": 3, "n_angular": 26, "C": C}
    if k <= 6:
        return {"ell_max": 4, "n_modes": 2, "K": 14, "n_radial": 5, "n_angular": 38, "C": C}
    if k <= 8:
        return {"ell_max": 5, "n_modes": 3, "K": 16, "n_radial": 5, "n_angular": 50, "C": C}
    if k <= 9:
        return {"ell_max": 6, "n_modes": 3, "K": 18, "n_radial": 5, "n_angular": 74, "C": C}
    return {"ell_max": 7, "n_modes": 3, "K": 20, "n_radial": 6, "n_angular": 74, "C": C}


def run_experiment(config: ExperimentConfig) -> Any:
    requested_measure_dirs = 110
    grid_size = 81
    noise_level = 0.2
    epsilon = 0.2
    kind = "full"
    k_values = [4, 6, 7, 8, 9, 10]
    quad_order = 160
    r_eval_count = 120
    half_side = 1.0
    bandwidth_factor = 2.0
    # Empirical stabilization for Fourier/Bessel data least squares.
    basis_lstsq_rcond = 1e-7
    fourier_mode_fraction = 1.5
    bessel_mode_fraction = 1.1
    basis_mode_min = 12

    if config.quick:
        requested_measure_dirs = 38
        grid_size = 51
        k_values = [4, 6, 8, 10]
        quad_order = 100
        r_eval_count = 80

    rng = np.random.default_rng(config.seed + 600)
    data_mode = getattr(config, "data_mode", "mock")
    blocks = three_tensor_block_phantom()
    n_tensor_coeffs = len(tensor_basis(kind))

    n_rows = len(k_values)
    n_cols = 6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.1 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[None, :]

    diagnostic_rows: list[dict[str, Any]] = []

    for row_idx, k in enumerate(k_values):
        rp = _row_params_quick(k) if config.quick else _row_params(k)
        C = float(rp["C"])
        ell_max = int(rp["ell_max"])
        n_modes = int(rp["n_modes"])
        K_val = int(rp["K"])
        n_radial = int(rp["n_radial"])
        n_angular = int(rp["n_angular"])

        target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)
        p_nodes, _, _, mock_distances, _ = generate_data_nodes(
            target_nodes,
            requested_measure_dirs,
            data_mode=data_mode,
            branch_count=1,
        )

        alpha_df = collect_alpha_pairs_cached(
            C,
            K_val,
            ell_max,
            n_modes,
            quad_order=quad_order,
            r_eval_count=r_eval_count,
            cache_dir=config.out_dir / "alpha_cache",
        )
        alpha_lookup = {
            (int(r["ell"]), int(r["n"])): complex(float(r["alpha_real"]), float(r["alpha_imag"]))
            for _, r in alpha_df.iterrows()
        }
        modes: list[Mode] = []
        for ell in range(ell_max + 1):
            _, beta = solve_ball_gpswf(C, ell, K_val, n_modes=n_modes)
            for n in range(beta.shape[1]):
                alpha = alpha_lookup[(ell, n)]
                for m in range(-ell, ell + 1):
                    modes.append(Mode(ell=ell, n=n, m=m, alpha=alpha, beta=beta[:, n]))

        alpha_abs = np.asarray([abs(mode.alpha) for mode in modes], dtype=float)
        retained = alpha_abs > epsilon * float(np.max(alpha_abs))
        N_cap = int(C * C / 2)
        if np.sum(retained) > N_cap:
            order = np.argsort(-alpha_abs)
            keep = order[:N_cap]
            retained = np.zeros(len(modes), dtype=bool)
            retained[keep] = True
        fourier_mode_cap = max(
            int(basis_mode_min),
            int(fourier_mode_fraction * int(np.sum(retained))),
        )
        bessel_mode_cap = max(
            int(basis_mode_min),
            int(bessel_mode_fraction * int(np.sum(retained))),
        )

        target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
        xs = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(xs, xs)
        grid_points = np.column_stack([
            X.reshape(-1),
            Y.reshape(-1),
            np.zeros(grid_size * grid_size),
        ])
        image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

        truth, _, disk_mask = tensor_truth_image_2d(
            grid_size,
            blocks,
            kind=kind,
            display="frobenius",
        )
        vmin = 0.0
        vmax = float(np.nanmax(np.real(truth)))
        _imshow(axes[row_idx, 0], np.real(truth), "truth ||Q||F" if row_idx == 0 else "", vmin, vmax)

        true_coeffs = tensor_block_fourier_coefficients(p_nodes, blocks, C, kind)
        farfield_data, sigma_min, cond_values = polarimetric_farfield_data(
            p_nodes,
            true_coeffs,
            kind,
            noise_level,
            rng,
        )
        recovered = recover_polarimetric_coefficients_from_data(
            p_nodes,
            farfield_data,
            kind,
        )
        data_nodes = p_nodes
        data_weights = target_weights
        em_farfield_data = farfield_data
        if data_mode == "ideal":
            recovered = recovered.reshape(-1, target_nodes.shape[0], n_tensor_coeffs).mean(axis=0)
            em_farfield_data = em_farfield_data.reshape(
                -1,
                target_nodes.shape[0],
                em_farfield_data.shape[1],
            ).mean(axis=0)
            data_nodes = target_nodes

        gpswf_components = np.empty((grid_points.shape[0], n_tensor_coeffs), dtype=np.complex128)
        coeff_norms = []
        for component_index in range(n_tensor_coeffs):
            coeffs = quadrature_modal_coefficients(
                recovered[:, component_index],
                target_basis,
                target_weights,
                modes,
                retained,
            )
            coeff_norms.append(vector_norm(coeffs))
            gpswf_components[:, component_index] = image_matrix @ coeffs
        gpswf_image = np.linalg.norm(gpswf_components, axis=1).reshape(grid_size, grid_size)
        gpswf_image[~disk_mask] = 0.0
        _imshow(axes[row_idx, 1], gpswf_image, "GPSWF ||Q||F" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_tensor_diagnostics(
            case={
                "figure": 6,
                "case_id": f"k{k:g}_gpswf_tensor",
                "method": "gpswf_tensor_frobenius",
                "row": int(row_idx),
                "column": 1,
                "k": float(k),
                "C": float(C),
                "K": int(K_val),
                "ell_max": int(ell_max),
                "n_modes_per_ell": int(n_modes),
                "epsilon": float(epsilon),
                "N_cap": int(N_cap),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "retained_modes": int(np.sum(retained)),
                "total_modes": int(len(modes)),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "data_norm": vector_norm(recovered),
                "coeff_norm": float(np.linalg.norm(coeff_norms)),
                "polarimetric_sigma_min_median": float(np.median(sigma_min)),
                "polarimetric_cond_median": float(np.median(cond_values)),
            },
            image=gpswf_image,
            truth=np.real(truth),
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        fourier_components = np.empty((grid_points.shape[0], n_tensor_coeffs), dtype=np.complex128)
        fourier_coeff_norms = []
        fourier_meta: dict[str, Any] = {}
        for component_index in range(n_tensor_coeffs):
            values, meta = reconstruct_fourier_cube_from_data(
                recovered[:, component_index],
                data_nodes,
                data_weights,
                grid_points,
                float(k),
                C,
                half_side=half_side,
                bandwidth_factor=bandwidth_factor,
                max_modes=fourier_mode_cap,
                rcond=basis_lstsq_rcond,
            )
            fourier_components[:, component_index] = values
            fourier_coeff_norms.append(float(meta["coeff_norm"]))
            if component_index == 0:
                fourier_meta = meta
        fourier_image = np.linalg.norm(fourier_components, axis=1).reshape(grid_size, grid_size)
        _imshow(axes[row_idx, 2], fourier_image, "Cube Fourier ||Q||F" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_tensor_diagnostics(
            case={
                "figure": 6,
                "case_id": f"k{k:g}_fourier_tensor",
                "method": "cube_fourier_tensor_lstsq",
                "row": int(row_idx),
                "column": 2,
                "k": float(k),
                "C": float(C),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "retained_modes": int(fourier_meta["fourier_modes"]),
                "total_modes": int(fourier_meta["fourier_candidate_modes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                **fourier_meta,
                "data_norm": vector_norm(recovered),
                "coeff_norm": float(np.linalg.norm(fourier_coeff_norms)),
                "projection_branch": "data_lstsq_capped",
                "basis_mode_fraction": float(fourier_mode_fraction),
                "basis_mode_cap": int(fourier_mode_cap),
                "polarimetric_sigma_min_median": float(np.median(sigma_min)),
                "polarimetric_cond_median": float(np.median(cond_values)),
            },
            image=fourier_image,
            truth=np.real(truth),
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        bessel_components = np.empty((grid_points.shape[0], n_tensor_coeffs), dtype=np.complex128)
        bessel_coeff_norms = []
        bessel_meta: dict[str, Any] = {}
        for component_index in range(n_tensor_coeffs):
            values, meta = reconstruct_ball_bessel_from_data(
                recovered[:, component_index],
                data_nodes,
                data_weights,
                grid_points,
                float(k),
                C,
                quadrature_nodes=target_nodes,
                quadrature_weights=target_weights,
                bandwidth_factor=bandwidth_factor,
                max_modes=bessel_mode_cap,
                rcond=basis_lstsq_rcond,
            )
            bessel_components[:, component_index] = values
            bessel_coeff_norms.append(float(meta["coeff_norm"]))
            if component_index == 0:
                bessel_meta = meta
        bessel_image = np.linalg.norm(bessel_components, axis=1).reshape(grid_size, grid_size)
        bessel_image[~disk_mask] = 0.0
        _imshow(axes[row_idx, 3], bessel_image, "Ball Bessel ||Q||F" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_tensor_diagnostics(
            case={
                "figure": 6,
                "case_id": f"k{k:g}_bessel_tensor",
                "method": "ball_bessel_tensor_lstsq",
                "row": int(row_idx),
                "column": 3,
                "k": float(k),
                "C": float(C),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "retained_modes": int(bessel_meta["bessel_modes"]),
                "total_modes": int(bessel_meta["bessel_candidate_modes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                **bessel_meta,
                "data_norm": vector_norm(recovered),
                "coeff_norm": float(np.linalg.norm(bessel_coeff_norms)),
                "projection_branch": "data_lstsq_capped",
                "basis_mode_fraction": float(bessel_mode_fraction),
                "basis_mode_cap": int(bessel_mode_cap),
                "polarimetric_sigma_min_median": float(np.median(sigma_min)),
                "polarimetric_cond_median": float(np.median(cond_values)),
            },
            image=bessel_image,
            truth=np.real(truth),
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        dsm_values, dsm_meta = direct_sampling_tensor_indicator(
            recovered,
            data_nodes,
            data_weights,
            grid_points,
            C,
            normalize=True,
        )
        dsm_image = dsm_values.reshape(grid_size, grid_size)
        dsm_image[~disk_mask] = 0.0
        dsm_display = np.real(dsm_image) * max(vmax, 1e-14)
        _imshow(axes[row_idx, 4], dsm_display, "DSM ||Q||F" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_tensor_diagnostics(
            case={
                "figure": 6,
                "case_id": f"k{k:g}_dsm_tensor",
                "method": "dsm_tensor_frobenius",
                "row": int(row_idx),
                "column": 4,
                "k": float(k),
                "C": float(C),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "retained_modes": int(dsm_meta["dsm_nodes"]),
                "total_modes": int(dsm_meta["dsm_nodes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "data_norm": float(dsm_meta["data_norm"]),
                "coeff_norm": float(dsm_meta["coeff_norm"]),
                "polarimetric_sigma_min_median": float(np.median(sigma_min)),
                "polarimetric_cond_median": float(np.median(cond_values)),
                **dsm_meta,
            },
            image=dsm_display,
            truth=np.real(truth),
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        em_dsm_values, em_dsm_meta = direct_sampling_farfield_indicator(
            em_farfield_data,
            data_nodes,
            data_weights,
            grid_points,
            C,
            kind=kind,
            normalize=True,
        )
        em_dsm_image = em_dsm_values.reshape(grid_size, grid_size)
        em_dsm_image[~disk_mask] = 0.0
        em_dsm_display = np.real(em_dsm_image) * max(vmax, 1e-14)
        _imshow(axes[row_idx, 5], em_dsm_display, "EM-DSM" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_tensor_diagnostics(
            case={
                "figure": 6,
                "case_id": f"k{k:g}_em_dsm_tensor",
                "method": "em_dsm_farfield_channels",
                "row": int(row_idx),
                "column": 5,
                "k": float(k),
                "C": float(C),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "retained_modes": int(em_dsm_meta["em_dsm_nodes"]),
                "total_modes": int(em_dsm_meta["em_dsm_nodes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                "data_norm": float(em_dsm_meta["farfield_data_norm"]),
                "coeff_norm": float(em_dsm_meta["farfield_weighted_norm"]),
                "polarimetric_sigma_min_median": float(np.median(sigma_min)),
                "polarimetric_cond_median": float(np.median(cond_values)),
                "projection_branch": "raw_farfield_phase_backprojection",
                **em_dsm_meta,
            },
            image=em_dsm_display,
            truth=np.real(truth),
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        axes[row_idx, 0].set_ylabel(
            (
                f"k={k:g}\n"
                f"GPSWF={int(np.sum(retained))}\n"
                f"Fourier={fourier_meta['fourier_modes']}\n"
                f"Bessel={bessel_meta['bessel_modes']}\n"
                f"DSM={dsm_meta['dsm_nodes']}\n"
                f"EM={em_dsm_meta['em_dsm_channels']}ch"
            ),
            fontsize=8,
            rotation=90,
            labelpad=12,
        )

    fig.savefig(config.out_dir / "figure6_tensor_blocks.png", dpi=200)
    plt.close(fig)
    write_diagnostics_csv(diagnostic_rows, config.out_dir / "figure6_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "figure6_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "figure6_diagnostic_curves.png",
        title="Figure 6 diagnostics",
    )
    print(f"Saved {config.out_dir / 'figure6_tensor_blocks.png'}")
    return make_table([{"figure": 6, "status": "ok"}])


def _tensor_diagnostics(
    *,
    case: dict[str, Any],
    image: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    mock_distances: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(case)
    img = np.asarray(image, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    target_mask = valid & (truth_arr > 1e-12)
    background_mask = valid & ~target_mask

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(img[mask])) if np.any(mask) else float("nan")

    def p95(mask: np.ndarray) -> float:
        return float(np.percentile(img[mask], 95)) if np.any(mask) else float("nan")

    if mock_distances is None:
        mock_mean = mock_max = mock_p95 = float("nan")
    else:
        distances = np.asarray(mock_distances, dtype=float)
        mock_mean = float(np.mean(distances)) if distances.size else float("nan")
        mock_max = float(np.max(distances)) if distances.size else float("nan")
        mock_p95 = float(np.percentile(distances, 95)) if distances.size else float("nan")

    row.update({
        "mock_distance_mean": mock_mean,
        "mock_distance_max": mock_max,
        "mock_distance_p95": mock_p95,
        "image_min": float(np.min(img)),
        "image_max": float(np.max(img)),
        "image_max_abs": float(np.max(np.abs(img))),
        "image_l2": vector_norm(img),
        "background_mean_abs": mean(background_mask),
        "background_p95_abs": p95(background_mask),
        "target_mean_abs": mean(target_mask),
        "target_p95_abs": p95(target_mask),
    })
    return row


def _imshow(ax, img, title, vmin, vmax):
    ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-mode", choices=["mock", "ideal"], default="mock")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(
        ExperimentConfig(
            out_dir=out_dir,
            seed=args.seed,
            quick=args.quick,
            data_mode=args.data_mode,
        )
    )


if __name__ == "__main__":
    main()
