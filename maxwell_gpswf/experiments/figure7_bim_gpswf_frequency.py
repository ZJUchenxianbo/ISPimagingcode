#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 7: BIM-GPSWF frequency experiment.

Analytic Born + Full VIE columns use the unified polarimetric pipeline.
BIM iterations use the same normalized raw Full VIE far-field channels as the
Full VIE GPSWF initial image.  The contrast model is ``Q(x)=q(x)T0`` with
known tensor T0, and the update is restricted to retained GPSWF modes.
"""
from __future__ import annotations

import argparse
import math
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
    generate_data_nodes,
    make_table,
    modal_matrix,
    plot_diagnostic_curves,
    quadrature_modal_coefficients,
    recover_polarimetric_coefficients,
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
from forward.datasets import (
    FarfieldDataset,
    farfield_dataset_to_qhat,
)
from forward.vie import ball_voxel_grid
from nonlinear import (
    compute_raw_bim_gpswf_linearization,
    compute_raw_vie_farfield_data,
    evaluate_blocks_on_nodes,
    solve_tikhonov_update,
)


def _row_params(k: float) -> dict[str, float | int]:
    """GPSWF parameters copied from Figure 5 for the same k rows."""
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
    k_values = [5, 6, 7, 8, 9, 10]
    n_per_axis = 7
    n_iterations = 3
    epsilon = 0.2
    step_size = 0.2
    lambda0 = 1e-2
    quad_order = 160
    r_eval_count = 120
    data_mode = getattr(config, "data_mode", "mock")
    kind = "isotropic"

    if config.quick:
        requested_measure_dirs = 38
        grid_size = 51
        k_values = [4, 6, 8, 10]
        n_per_axis = 5
        quad_order = 100
        r_eval_count = 80

    tensor = reference_tensor(kind)
    comp_scale = complex(tensor_coefficients_from_matrix(tensor, kind)[0])
    blocks = three_block_phantom("born")

    volume_nodes, volume_weights, voxel_h = ball_voxel_grid(1.0, n_per_axis)
    q_true_values = evaluate_blocks_on_nodes(volume_nodes, blocks)

    n_rows = len(k_values)
    titles = [
        "truth",
        "Analytic Born FF",
        "Full VIE data",
        "BIM iter 1",
        "BIM iter 2",
        "BIM iter 3",
    ]
    n_cols = len(titles)
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
        n_modes_per_ell = int(rp["n_modes"])
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
        # VIE far-field data uses the exp(+i C p.x) phase.  To obtain data in
        # the project exp(-i C p.x) convention for the positive quadrature node
        # p, construct the physical incident/observation pairs for -p.
        vie_p_nodes, inc_dirs, obs_dirs, vie_mock_distances, _ = generate_data_nodes(
            -target_nodes,
            requested_measure_dirs,
            data_mode=data_mode,
            branch_count=1,
        )
        vie_reconstruction_nodes = -vie_p_nodes

        modes = _build_modes(
            C=C,
            K=K_val,
            ell_max=ell_max,
            n_modes_per_ell=n_modes_per_ell,
            quad_order=quad_order,
            r_eval_count=r_eval_count,
            cache_dir=config.out_dir / "alpha_cache",
        )
        alpha_abs = np.asarray([abs(mode.alpha) for mode in modes], dtype=float)
        retained = alpha_abs > epsilon * float(np.max(alpha_abs))
        N_cap = int(C * C / 2)
        if int(np.sum(retained)) > N_cap:
            order = np.argsort(-alpha_abs)
            keep = order[:N_cap]
            retained = np.zeros(len(modes), dtype=bool)
            retained[keep] = True

        target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
        volume_basis = modal_matrix(volume_nodes, modes, fourier_side=False)
        retained_volume_basis = volume_basis[:, retained]

        xs = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(xs, xs)
        grid_points = np.column_stack([
            X.reshape(-1),
            Y.reshape(-1),
            np.zeros(grid_size * grid_size),
        ])
        image_matrix = modal_matrix(grid_points, modes, fourier_side=False)

        truth, _, disk_mask = truth_image_2d(grid_size, blocks, comp_scale)
        vmin = float(np.nanmin(np.real(truth)))
        vmax = float(np.nanmax(np.real(truth)))
        _imshow(axes[row_idx, 0], np.real(truth), titles[0] if row_idx == 0 else "", vmin, vmax)

        # Analytic Born: full pipeline (like figs 1/2)
        coeff0 = tensor_coefficients_from_matrix(tensor, kind)
        scalar_born = block_fourier_profile(p_nodes, blocks, C)
        tc_born = scalar_born[:, None] * coeff0[None, :]
        rec_c_born, _, _ = recover_polarimetric_coefficients(
            p_nodes, tc_born, kind, 0.0,
            np.random.default_rng(config.seed + 700))
        comp_born = rec_c_born[:, 0]

        # Full VIE observation: one raw far-field dataset feeds both the
        # polarimetric GPSWF initial image and the BIM residual.
        observed = compute_raw_vie_farfield_data(
            incident_dirs=inc_dirs,
            obs_dirs=obs_dirs,
            volume_nodes=volume_nodes,
            volume_weights=volume_weights,
            q_values=q_true_values,
            tensor=tensor,
            k=float(k),
            h=voxel_h,
            return_fields=False,
            conjugate_to_unified=False,
        )
        observed_dataset = FarfieldDataset(
            p_nodes=vie_reconstruction_nodes,
            incident_dirs=inc_dirs,
            obs_dirs=obs_dirs,
            farfield_data=observed.farfield_data,
            data_source="full_vie",
            metadata={
                "kind": kind,
                "k": float(k),
                "farfield_normalization": "M_c",
                "fourier_convention": "exp(-i C p.x)",
                "physical_node_sign": -1,
            },
        )
        rec_c_vie = farfield_dataset_to_qhat(
            observed_dataset, kind=kind, noise_level=0.0,
            rng=np.random.default_rng(config.seed + 701))
        comp_vie = rec_c_vie[:, 0]
        observed_vector = observed.farfield_data.reshape(-1)

        coeffs_analytic_born = quadrature_modal_coefficients(
            comp_born,
            target_basis,
            target_weights,
            modes,
            retained,
        )
        image_analytic_born = (image_matrix @ coeffs_analytic_born).reshape(grid_size, grid_size)
        image_analytic_born[~disk_mask] = 0.0
        _imshow(
            axes[row_idx, 1],
            np.real(image_analytic_born),
            titles[1] if row_idx == 0 else "",
            vmin,
            vmax,
        )
        diagnostic_rows.append(_diagnostic_row(
            figure=7,
            case_id=f"k{k:g}_analytic_born_farfield",
            method="analytic_born_farfield_gpswf",
            row=row_idx,
            column=1,
            k=float(k),
            iteration=0,
            image=image_analytic_born,
            truth=truth,
            disk_mask=disk_mask,
            retained_modes=int(np.sum(retained)),
            total_modes=len(modes),
            data_residual_norm=math.nan,
            relative_data_residual=math.nan,
            update_norm=math.nan,
            step_size=math.nan,
            lambda_value=math.nan,
            linear_solve_residual=math.nan,
            linear_solve_rank=-1,
            linear_solve_cond=math.nan,
            vie_matrix_residual=math.nan,
            bim_matrix_norm=math.nan,
            bim_matrix_cond=math.nan,
            data_mode=data_mode,
            bim_residual_space="none",
        ))

        coeffs_full_component = quadrature_modal_coefficients(
            comp_vie,
            target_basis,
            target_weights,
            modes,
            retained,
        )
        image_full_initial = (image_matrix @ coeffs_full_component).reshape(grid_size, grid_size)
        image_full_initial[~disk_mask] = 0.0
        _imshow(
            axes[row_idx, 2],
            np.real(image_full_initial),
            titles[2] if row_idx == 0 else "",
            vmin,
            vmax,
        )

        q_coeffs = coeffs_full_component / comp_scale
        current_scalar = volume_basis @ q_coeffs
        current_data = compute_raw_vie_farfield_data(
            incident_dirs=inc_dirs,
            obs_dirs=obs_dirs,
            volume_nodes=volume_nodes,
            volume_weights=volume_weights,
            q_values=current_scalar,
            tensor=tensor,
            k=float(k),
            h=voxel_h,
            return_fields=True,
            conjugate_to_unified=False,
        )
        residual = observed_vector - current_data.farfield_data.reshape(-1)
        initial_residual_norm = vector_norm(residual)
        diagnostic_rows.append(_diagnostic_row(
            figure=7,
            case_id=f"k{k:g}_full_vie_data_gpswf",
            method="full_vie_data_gpswf",
            row=row_idx,
            column=2,
            k=float(k),
            iteration=0,
            image=image_full_initial,
            truth=truth,
            disk_mask=disk_mask,
            retained_modes=int(np.sum(retained)),
            total_modes=len(modes),
            data_residual_norm=initial_residual_norm,
            relative_data_residual=initial_residual_norm / max(vector_norm(observed_vector), 1e-14),
            update_norm=0.0,
            step_size=0.0,
            lambda_value=math.nan,
            linear_solve_residual=math.nan,
            linear_solve_rank=-1,
            linear_solve_cond=math.nan,
            vie_matrix_residual=current_data.matrix_residual,
            bim_matrix_norm=math.nan,
            bim_matrix_cond=math.nan,
            data_mode=data_mode,
            bim_residual_space="raw_farfield_channel",
            data_residual_norm_before_update=initial_residual_norm,
            relative_data_residual_before_update=initial_residual_norm / max(vector_norm(observed_vector), 1e-14),
            mock_distance_mean=float(np.mean(mock_distances)) if mock_distances.size else math.nan,
            vie_mock_distance_mean=float(np.mean(vie_mock_distances)) if vie_mock_distances.size else math.nan,
        ))

        for iteration in range(1, n_iterations + 1):
            if current_data.total_fields is None:
                raise RuntimeError("BIM iteration requires total fields")
            linearization = compute_raw_bim_gpswf_linearization(
                incident_dirs=inc_dirs,
                obs_dirs=obs_dirs,
                volume_nodes=volume_nodes,
                volume_weights=volume_weights,
                total_fields=current_data.total_fields,
                retained_mode_values=retained_volume_basis,
                tensor=tensor,
                k=float(k),
                conjugate_to_unified=False,
            )
            residual_before_update = residual
            residual_before_norm = vector_norm(residual_before_update)
            update_retained, solve_meta = solve_tikhonov_update(
                linearization.matrix,
                residual_before_update,
                lambda0=lambda0,
            )
            update_coeffs = np.zeros(len(modes), dtype=np.complex128)
            update_coeffs[retained] = update_retained
            q_coeffs = q_coeffs + float(step_size) * update_coeffs

            current_scalar = volume_basis @ q_coeffs
            current_data = compute_raw_vie_farfield_data(
                incident_dirs=inc_dirs,
                obs_dirs=obs_dirs,
                volume_nodes=volume_nodes,
                volume_weights=volume_weights,
                q_values=current_scalar,
                tensor=tensor,
                k=float(k),
                h=voxel_h,
                return_fields=True,
                conjugate_to_unified=False,
            )
            residual = observed_vector - current_data.farfield_data.reshape(-1)
            residual_after_norm = vector_norm(residual)

            image_bim = (image_matrix @ (q_coeffs * comp_scale)).reshape(grid_size, grid_size)
            image_bim[~disk_mask] = 0.0
            _imshow(
                axes[row_idx, iteration + 2],
                np.real(image_bim),
                titles[iteration + 2] if row_idx == 0 else "",
                vmin,
                vmax,
            )
            diagnostic_rows.append(_diagnostic_row(
                figure=7,
                case_id=f"k{k:g}_bim_iter{iteration}",
                method="bim_gpswf",
                row=row_idx,
                column=iteration + 2,
                k=float(k),
                iteration=iteration,
                image=image_bim,
                truth=truth,
                disk_mask=disk_mask,
                retained_modes=int(np.sum(retained)),
                total_modes=len(modes),
                data_residual_norm=residual_after_norm,
                relative_data_residual=residual_after_norm / max(vector_norm(observed_vector), 1e-14),
                update_norm=float(solve_meta["update_norm"]),
                step_size=float(step_size),
                lambda_value=float(solve_meta["lambda"]),
                linear_solve_residual=float(solve_meta["linear_solve_residual"]),
                linear_solve_rank=int(solve_meta["linear_solve_rank"]),
                linear_solve_cond=float(solve_meta["linear_solve_cond"]),
                vie_matrix_residual=current_data.matrix_residual,
                bim_matrix_norm=linearization.matrix_norm,
                bim_matrix_cond=linearization.condition,
                data_mode=data_mode,
                bim_residual_space="raw_farfield_channel",
                data_residual_norm_before_update=residual_before_norm,
                relative_data_residual_before_update=residual_before_norm / max(vector_norm(observed_vector), 1e-14),
                mock_distance_mean=float(np.mean(mock_distances)) if mock_distances.size else math.nan,
                vie_mock_distance_mean=float(np.mean(vie_mock_distances)) if vie_mock_distances.size else math.nan,
            ))

        axes[row_idx, 0].set_ylabel(
            (
                f"k={k:g}\n"
                f"modes={int(np.sum(retained))}\n"
                f"vox={volume_nodes.shape[0]}"
            ),
            fontsize=8,
            rotation=90,
            labelpad=12,
        )

    fig.savefig(config.out_dir / "figure7_bim_gpswf_frequency.png", dpi=200)
    plt.close(fig)
    write_diagnostics_csv(diagnostic_rows, config.out_dir / "figure7_diagnostics.csv")
    save_diagnostics_npz(diagnostic_rows, config.out_dir / "figure7_diagnostics_detail.npz")
    _plot_residual_curves(diagnostic_rows, config.out_dir / "figure7_residual_curves.png")
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / "figure7_diagnostic_curves.png",
        title="Figure 7 diagnostics",
    )
    print(f"Saved {config.out_dir / 'figure7_bim_gpswf_frequency.png'}")
    return make_table([{"figure": 7, "status": "ok"}])


def _build_modes(
    *,
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    quad_order: int,
    r_eval_count: int,
    cache_dir: Path,
) -> list[Mode]:
    alpha_df = collect_alpha_pairs_cached(
        C,
        K,
        ell_max,
        n_modes_per_ell,
        quad_order=quad_order,
        r_eval_count=r_eval_count,
        cache_dir=cache_dir,
    )
    alpha_lookup = {
        (int(r["ell"]), int(r["n"])): complex(float(r["alpha_real"]), float(r["alpha_imag"]))
        for _, r in alpha_df.iterrows()
    }
    modes: list[Mode] = []
    for ell in range(ell_max + 1):
        _, beta = solve_ball_gpswf(C, ell, K, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            alpha = alpha_lookup[(ell, n)]
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=alpha, beta=beta[:, n]))
    return modes


def _diagnostic_row(
    *,
    figure: int,
    case_id: str,
    method: str,
    row: int,
    column: int,
    k: float,
    iteration: int,
    image: np.ndarray,
    truth: np.ndarray,
    disk_mask: np.ndarray,
    retained_modes: int,
    total_modes: int,
    data_residual_norm: float,
    relative_data_residual: float,
    update_norm: float,
    step_size: float,
    lambda_value: float,
    linear_solve_residual: float,
    linear_solve_rank: int,
    linear_solve_cond: float,
    vie_matrix_residual: float,
    bim_matrix_norm: float,
    bim_matrix_cond: float,
    data_mode: str,
    data_residual_norm_before_update: float = math.nan,
    relative_data_residual_before_update: float = math.nan,
    mock_distance_mean: float = math.nan,
    vie_mock_distance_mean: float = math.nan,
    bim_residual_space: str = "",
) -> dict[str, Any]:
    abs_img = np.abs(image)
    valid = np.asarray(disk_mask, dtype=bool)
    target_mask = valid & (np.abs(truth) > 1e-12)
    background_mask = valid & ~target_mask

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_img[mask])) if np.any(mask) else math.nan

    def p95(mask: np.ndarray) -> float:
        return float(np.percentile(abs_img[mask], 95)) if np.any(mask) else math.nan

    target_p95 = p95(target_mask)
    background_p95 = p95(background_mask)
    return {
        "figure": int(figure),
        "case_id": case_id,
        "method": method,
        "row": int(row),
        "column": int(column),
        "k": float(k),
        "iteration": int(iteration),
        "data_mode": data_mode,
        "bim_residual_space": bim_residual_space,
        "retained_modes": int(retained_modes),
        "total_modes": int(total_modes),
        "data_residual_norm": float(data_residual_norm),
        "relative_data_residual": float(relative_data_residual),
        "data_residual_norm_before_update": float(data_residual_norm_before_update),
        "relative_data_residual_before_update": float(relative_data_residual_before_update),
        "update_norm": float(update_norm),
        "step_size": float(step_size),
        "lambda": float(lambda_value),
        "linear_solve_residual": float(linear_solve_residual),
        "linear_solve_rank": int(linear_solve_rank),
        "linear_solve_cond": float(linear_solve_cond),
        "vie_relative_residual": float(vie_matrix_residual),
        "bim_matrix_norm": float(bim_matrix_norm),
        "bim_matrix_cond": float(bim_matrix_cond),
        "mock_distance_mean": float(mock_distance_mean),
        "vie_mock_distance_mean": float(vie_mock_distance_mean),
        "image_min": float(np.min(np.real(image))),
        "image_max": float(np.max(np.real(image))),
        "image_max_abs": float(np.max(abs_img)),
        "target_mean_abs": mean(target_mask),
        "target_p95_abs": target_p95,
        "background_mean_abs": mean(background_mask),
        "background_p95_abs": background_p95,
        "target_background_ratio": target_p95 / max(background_p95, 1e-14),
    }


def _plot_residual_curves(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    k_values = sorted({float(row["k"]) for row in rows if row["method"] == "bim_gpswf"})
    for k in k_values:
        selected = [
            row for row in rows
            if row["method"] == "bim_gpswf" and abs(float(row["k"]) - k) < 1e-12
        ]
        selected.sort(key=lambda row: int(row["iteration"]))
        if not selected:
            continue
        x = [int(row["iteration"]) for row in selected]
        y = [float(row["relative_data_residual"]) for row in selected]
        ax.plot(x, y, marker="o", label=f"k={k:g}")
    ax.set_xlabel("BIM iteration")
    ax.set_ylabel("relative data residual before update")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _imshow(ax, img, title, vmin, vmax):
    ax.imshow(img, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="outputs/fig7")
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
