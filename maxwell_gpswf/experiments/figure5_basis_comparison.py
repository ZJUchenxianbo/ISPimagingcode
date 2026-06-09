#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5: GPSWF, cube Fourier, and ball Bessel basis comparison.

Rows use the current Figure 2 wave-number configuration.  For each row, GPSWF
uses single-frequency Born data at ``k`` on the unit ball, while the Fourier
baseline uses cube Fourier modes on ``[-1,1]^3`` with ``|xi| <= 2*K_max`` and
``K_max = k``.  The ball Bessel baseline uses the unit-ball Dirichlet basis
with ``rho_{ell,n} <= 2*K_max``.  All three methods use the same recovered
polarimetric Fourier data; only the final reconstruction basis is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    Block,
    ExperimentConfig,
    ball_quadrature_nodes,
    block_fourier_profile,
    collect_alpha_pairs_cached,
    collect_reconstruction_diagnostics,
    generate_data_nodes,
    make_table,
    modal_matrix,
    plot_diagnostic_curves,
    quadrature_modal_coefficients,
    recover_polarimetric_coefficients,
    reconstruct_ball_bessel_from_data,
    reconstruct_fourier_cube_from_data,
    reference_tensor,
    save_diagnostics_npz,
    solve_ball_gpswf,
    tensor_coefficients_from_matrix,
    three_block_phantom,
    truth_image_2d,
    write_diagnostics_csv,
)
from common.phantom import Mode
from common.utils import vector_norm


def _row_params(k: float) -> dict[str, float | int]:
    """GPSWF parameters copied from the current Figure 2 configuration."""
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
    component_index = 0
    contrast_scale = 1.0
    k_values = [4, 6, 7, 8, 9, 10]
    quad_order = 160
    r_eval_count = 120
    half_side = 1.0
    bandwidth_factor = 2.0

    if config.quick:
        requested_measure_dirs = 38
        grid_size = 51
        k_values = [4, 6, 8, 10]
        quad_order = 100
        r_eval_count = 80

    rng = np.random.default_rng(config.seed + 500)
    data_mode = getattr(config, "data_mode", "mock")
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)
    base_blocks = three_block_phantom("born")

    n_rows = len(k_values)
    # truth + GPSWF + cube Fourier + ball Bessel, all at medium contrast
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.1 * n_rows),
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

        target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
        xs = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(xs, xs)
        grid_points = np.column_stack([
            X.reshape(-1),
            Y.reshape(-1),
            np.zeros(grid_size * grid_size),
        ])
        image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

        truth, _, disk_mask = truth_image_2d(grid_size, base_blocks, coeff0[component_index])
        square_mask = np.ones_like(disk_mask, dtype=bool)
        vmin = float(np.nanmin(np.real(truth)))
        vmax = float(np.nanmax(np.real(truth)))
        _imshow(axes[row_idx, 0], np.real(truth), "truth" if row_idx == 0 else "", vmin, vmax)

        scaled_blocks = [
            Block(
                center=block.center,
                half_width=block.half_width,
                amplitude=complex(
                    block.amplitude.real * contrast_scale,
                    block.amplitude.imag * contrast_scale,
                ),
            )
            for block in base_blocks
        ]

        scalar = block_fourier_profile(p_nodes, scaled_blocks, C=C)
        true_coeffs = scalar[:, None] * coeff0[None, :]
        recovered, _, _ = recover_polarimetric_coefficients(
            p_nodes,
            true_coeffs,
            kind,
            noise_level,
            rng,
        )
        comp_data = recovered[:, component_index]
        if data_mode == "ideal":
            comp_data = comp_data.reshape(-1, target_nodes.shape[0]).mean(axis=0)
        gpswf_coeffs = quadrature_modal_coefficients(
            comp_data,
            target_basis,
            target_weights,
            modes,
            retained,
        )
        gpswf_rec = (image_matrix @ gpswf_coeffs).reshape(grid_size, grid_size)
        gpswf_rec[~disk_mask] = 0.0
        _imshow(axes[row_idx, 1], np.real(gpswf_rec), "GPSWF" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case={
                "figure": 5,
                "case_id": f"k{k:g}_gpswf_medium",
                "method": "gpswf_single_frequency",
                "row": int(row_idx),
                "column": 1,
                "k": float(k),
                "K_max": float(k),
                "C": float(C),
                "K": int(K_val),
                "ell_max": int(ell_max),
                "n_modes_per_ell": int(n_modes),
                "epsilon": float(epsilon),
                "N_cap": int(N_cap),
                "n_radial": int(n_radial),
                "n_angular_requested": int(n_angular),
                "data_mode": data_mode,
                "noise_level": float(noise_level),
                "contrast_scale": float(contrast_scale),
                "support": "unit_ball",
                "data_source": "analytical_born",
            },
            modes=modes,
            retained=retained,
            target_nodes=target_nodes,
            p_nodes=p_nodes,
            mock_distances=mock_distances,
            basis_matrix=target_basis,
            target_weights=target_weights,
            component_data=comp_data,
            coeffs=gpswf_coeffs,
            image=gpswf_rec,
            truth=truth,
            disk_mask=disk_mask,
        ))

        data_nodes = target_nodes if data_mode == "ideal" else p_nodes

        fourier_values, fourier_meta = reconstruct_fourier_cube_from_data(
            comp_data,
            data_nodes,
            target_weights,
            grid_points,
            float(k),
            C,
            half_side=half_side,
            bandwidth_factor=bandwidth_factor,
        )
        fourier_rec = fourier_values.reshape(grid_size, grid_size)
        _imshow(axes[row_idx, 2], np.real(fourier_rec), "Cube Fourier" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_baseline_diagnostics(
            case={
                "figure": 5,
                "case_id": f"k{k:g}_fourier_medium",
                "method": "cube_fourier_multifrequency",
                "row": int(row_idx),
                "column": 2,
                "k": float(k),
                "K_max": float(k),
                "C": float(C),
                "noise_level": float(noise_level),
                "contrast_scale": float(contrast_scale),
                "support": "cube_half_side_1",
                "projection_branch": "data_lstsq",
                "basis_modes": int(fourier_meta["fourier_modes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                **fourier_meta,
            },
            image=fourier_rec,
            truth=truth,
            valid_mask=square_mask,
            mock_distances=mock_distances,
        ))

        bessel_values, bessel_meta = reconstruct_ball_bessel_from_data(
            comp_data,
            data_nodes,
            target_weights,
            grid_points,
            float(k),
            C,
            quadrature_nodes=target_nodes,
            quadrature_weights=target_weights,
            bandwidth_factor=bandwidth_factor,
        )
        bessel_rec = bessel_values.reshape(grid_size, grid_size)
        bessel_rec[~disk_mask] = 0.0
        _imshow(axes[row_idx, 3], np.real(bessel_rec), "Ball Bessel" if row_idx == 0 else "", vmin, vmax)
        diagnostic_rows.append(_baseline_diagnostics(
            case={
                "figure": 5,
                "case_id": f"k{k:g}_bessel_medium",
                "method": "ball_bessel_l2_projection",
                "row": int(row_idx),
                "column": 3,
                "k": float(k),
                "K_max": float(k),
                "C": float(C),
                "noise_level": float(noise_level),
                "contrast_scale": float(contrast_scale),
                "support": "unit_ball",
                "projection_branch": "data_lstsq",
                "basis_modes": int(bessel_meta["bessel_modes"]),
                "target_nodes": int(target_nodes.shape[0]),
                "p_nodes": int(data_nodes.shape[0]),
                **bessel_meta,
            },
            image=bessel_rec,
            truth=truth,
            valid_mask=disk_mask,
            mock_distances=mock_distances,
        ))

        axes[row_idx, 0].set_ylabel(
            (
                f"k/Kmax={k}\n"
                f"GPSWF={int(np.sum(retained))}\n"
                f"Fourier={fourier_meta['fourier_modes']}\n"
                f"Bessel={bessel_meta['bessel_modes']}"
            ),
            fontsize=8,
            rotation=90,
            labelpad=12,
        )

    fig.savefig(config.out_dir / "figure5_basis_comparison.png", dpi=200)
    plt.close(fig)
    write_diagnostics_csv(diagnostic_rows, config.out_dir / "figure5_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "figure5_diagnostics_detail.npz")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "figure5_diagnostic_curves.png",
        title="Figure 5 diagnostics",
    )
    print(f"Saved {config.out_dir / 'figure5_basis_comparison.png'}")
    return make_table([{"figure": 5, "status": "ok"}])


def _baseline_diagnostics(
    *,
    case: dict[str, Any],
    image: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    mock_distances: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(case)
    abs_img = np.abs(image)
    target_mask = valid_mask & (np.abs(truth) > 1e-12)
    background_mask = valid_mask & ~target_mask

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_img[mask])) if np.any(mask) else float("nan")

    def p95(mask: np.ndarray) -> float:
        return float(np.percentile(abs_img[mask], 95)) if np.any(mask) else float("nan")

    if mock_distances is None:
        mock_mean = mock_max = mock_p95 = float("nan")
    else:
        distances = np.asarray(mock_distances, dtype=float)
        mock_mean = float(np.mean(distances)) if distances.size else float("nan")
        mock_max = float(np.max(distances)) if distances.size else float("nan")
        mock_p95 = float(np.percentile(distances, 95)) if distances.size else float("nan")

    row.update({
        "retained_modes": int(row["basis_modes"]),
        "mock_distance_mean": mock_mean,
        "mock_distance_max": mock_max,
        "mock_distance_p95": mock_p95,
        "total_modes": int(row["basis_modes"]),
        "gram_offdiag_ratio": float("nan"),
        "gram_cond": float("nan"),
        "data_norm": float(row.get("data_norm", row["coeff_norm"])),
        "data_max_abs": float(row.get("data_max_abs", row["coeff_max_abs"])),
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
