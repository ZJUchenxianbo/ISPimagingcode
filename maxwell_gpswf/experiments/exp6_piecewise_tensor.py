#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 6: reconstruct a piecewise general complex tensor contrast.

Three disjoint blocks carry different, non-reciprocal ``3 x 3`` tensors.
The formal chain is

``Q(x) -> Full Maxwell VIE far field -> noise -> polarimetric Qhat -> GPSWF``.

All nine row-major tensor coefficients are reconstructed from the same finite
direction data, quadrature, retained GPSWF modes, and projection solve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig,
    ball_quadrature_nodes,
    build_gpswf_modes_with_chi,
    coefficient_diagnostics,
    generate_polarimetric_data_nodes,
    gpswf_reconstruct_tensor,
    modal_matrix,
    mode_diagnostics,
    node_diagnostics,
    projection_diagnostics,
    select_complete_gpswf_multiplets,
    tensor_coefficients_from_field,
    tensor_block_fourier_coefficients,
    tensor_material_diagnostics,
    tensor_matrix_from_coefficients,
    tensor_truth_matrix_2d,
    three_admissible_full_tensor_blocks,
    vector_norm,
    write_diagnostics_csv,
)
from common.basis_comparison import (
    basis_measure_dirs,
    basis_n_per_axis,
    basis_row_params,
    basis_row_params_quick,
)
from forward.datasets import (
    farfield_data_with_relative_noise,
    farfield_dataset_to_qhat,
    forward_solver_diagnostic_summary,
    full_vie_farfield_dataset,
    polarimetric_diagnostic_summary,
)


COMPONENT_LABELS = tuple(
    f"Q{row + 1}{column + 1}"
    for row in range(3)
    for column in range(3)
)


def _settings(quick: bool) -> tuple[int, int, int]:
    """Return image-grid, alpha-quadrature, and radial-evaluation sizes."""
    if quick:
        return 51, 100, 80
    return 161, 160, 120


def _relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    return vector_norm(values - reference) / max(vector_norm(reference), 1e-14)


def _block_masks(
    grid_points: np.ndarray,
    grid_size: int,
    blocks: list[Any],
) -> list[np.ndarray]:
    points = np.asarray(grid_points, dtype=float).reshape(grid_size, grid_size, 3)
    masks: list[np.ndarray] = []
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        masks.append(np.all(np.abs(points - center) <= half_width, axis=2))
    return masks


def _masked_panel(values: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    panel = np.asarray(values, dtype=float).copy()
    panel[~disk_mask] = 0.0
    return panel


def _plot_tensor_components(
    truth_coefficients: np.ndarray,
    reconstructed_coefficients: np.ndarray,
    disk_mask: np.ndarray,
    part: Callable[[np.ndarray], np.ndarray],
    part_name: str,
    component_indices: tuple[int, ...],
    path: Path,
) -> None:
    """Plot selected tensor components with one shared scale per component."""
    n_components = len(component_indices)
    fig, axes = plt.subplots(
        2,
        n_components,
        figsize=(3.8 * n_components, 6.2),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(2, n_components)
    truth_part = part(truth_coefficients)
    reconstruction_part = part(reconstructed_coefficients)
    for column, component in enumerate(component_indices):
        truth = _masked_panel(truth_part[:, :, component], disk_mask)
        reconstruction = _masked_panel(
            reconstruction_part[:, :, component],
            disk_mask,
        )
        valid_values = np.concatenate([
            truth[disk_mask].reshape(-1),
            reconstruction[disk_mask].reshape(-1),
        ])
        component_vmin = float(np.min(valid_values))
        component_vmax = float(np.max(valid_values))
        if component_vmax - component_vmin < 1e-12:
            component_vmin -= 0.5
            component_vmax += 0.5

        pair_axes = axes[:, column].tolist()
        last_image = None
        for row, (axis, panel) in enumerate(zip(pair_axes, (truth, reconstruction))):
            last_image = axis.imshow(
                panel,
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap="viridis",
                vmin=component_vmin,
                vmax=component_vmax,
                interpolation="bicubic",
            )
            if row == 0:
                label = COMPONENT_LABELS[component]
                axis.set_title(rf"${label[0]}_{{{label[1:]}}}$", fontsize=9)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
        if last_image is not None:
            fig.colorbar(
                last_image,
                ax=pair_axes,
                fraction=0.046,
                pad=0.03,
            )
    axes[0, 0].set_ylabel("Truth", fontsize=10)
    axes[1, 0].set_ylabel("GPSWF", fontsize=10)
    fig.suptitle(
        f"k = 12, noise = 0.2, selected {part_name} tensor components",
        fontsize=10,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_tensor_norm(
    truth_tensor: np.ndarray,
    reconstructed_tensor: np.ndarray,
    disk_mask: np.ndarray,
    path: Path,
) -> None:
    truth_norm = np.linalg.norm(truth_tensor, axis=(-2, -1))
    reconstruction_norm = np.linalg.norm(reconstructed_tensor, axis=(-2, -1))
    scale = max(float(np.max(truth_norm)), float(np.max(reconstruction_norm)), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
    panels = (
        (truth_norm, "Truth $\\|Q\\|_F$", "viridis", 0.0, scale),
        (reconstruction_norm, "GPSWF $\\|Q\\|_F$", "viridis", 0.0, scale),
    )
    last_image = None
    for axis, (panel, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            _masked_panel(panel, disk_mask),
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        last_image = image
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
    fig.suptitle("k = 12, noise = 0.2, full-tensor Frobenius norm", fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _complex_matrix_json(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return {
        "real": np.real(matrix).astype(float).tolist(),
        "imag": np.imag(matrix).astype(float).tolist(),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    grid_size, quad_order, r_eval_count = _settings(config.quick)

    k = 12.0
    C = 2.0 * k
    kind = "full"
    epsilon = 0.2
    noise_level = 0.2
    polarimetric_J = 6
    parameters = basis_row_params_quick(k) if config.quick else basis_row_params(k)
    ell_max = int(parameters["ell_max"])
    n_modes_per_ell = int(parameters["n_modes"])
    K = int(parameters["K"])
    n_radial = int(parameters["n_radial"])
    n_angular = int(parameters["n_angular"])
    n_per_axis = basis_n_per_axis(
        k,
        config.quick,
        experiment_number=4,
    )
    requested_measure_dirs = basis_measure_dirs(
        k,
        config.quick,
        experiment_number=4,
    )

    blocks = three_admissible_full_tensor_blocks()
    material_diagnostics = [
        tensor_material_diagnostics(block.tensor) for block in blocks
    ]

    target_nodes, target_weights, _ = ball_quadrature_nodes(
        n_radial,
        n_angular,
    )
    physical_nodes, incident_dirs, obs_dirs, mock_distances, data_info = (
        generate_polarimetric_data_nodes(
            -target_nodes,
            requested_measure_dirs,
            polarimetric_J=polarimetric_J,
            tensor_kind=kind,
        )
    )
    measurement_nodes = -physical_nodes

    dataset = full_vie_farfield_dataset(
        "piecewise_full_tensor_blocks",
        measurement_nodes,
        kind=kind,
        k=k,
        R=1.0,
        n_per_axis=n_per_axis,
        n_geometries=polarimetric_J,
        incident_dirs=incident_dirs,
        obs_dirs=obs_dirs,
    )
    rng = np.random.default_rng(config.seed + 600)
    standard_noise = (
        rng.normal(size=dataset.farfield_data.shape)
        + 1j * rng.normal(size=dataset.farfield_data.shape)
    )
    noisy_farfield = farfield_data_with_relative_noise(
        dataset,
        noise_level,
        standard_noise=standard_noise,
    )
    recovered_qhat = farfield_dataset_to_qhat(
        dataset,
        kind=kind,
        noise_level=noise_level,
        prepared_farfield_data=noisy_farfield,
    )
    # Diagnostic reference only: it is never used as reconstruction input.
    truth_qhat = tensor_block_fourier_coefficients(
        target_nodes,
        blocks,
        C,
        kind,
    )

    if int(dataset.metadata["polarimetric_rank_min"]) != 9:
        raise RuntimeError("Experiment 6 requires rank-9 polarimetric recovery")

    modes, chi_by_pair = build_gpswf_modes_with_chi(
        C,
        K,
        ell_max,
        n_modes_per_ell,
        quad_order,
        r_eval_count,
        config.out_dir / "alpha_cache",
    )
    alpha_abs = np.asarray([abs(mode.alpha) for mode in modes], dtype=float)
    eligible = alpha_abs > epsilon * float(np.max(alpha_abs))
    N_cap_theory = int(C * C / 2.0)
    N_cap_discrete = target_nodes.shape[0] // polarimetric_J
    N_cap_hard = 512
    N_cap = min(N_cap_theory, N_cap_discrete, N_cap_hard)
    retained, truncation_info = select_complete_gpswf_multiplets(
        modes,
        chi_by_pair,
        N_cap,
        eligible=eligible,
    )

    target_basis = modal_matrix(target_nodes, modes, fourier_side=True)
    truth_tensor, grid_points, disk_mask = tensor_truth_matrix_2d(
        grid_size,
        blocks,
    )
    truth_coefficients = tensor_coefficients_from_field(truth_tensor, kind)
    image_matrix = modal_matrix(grid_points, modes, fourier_side=False)
    reconstructed_coefficients, modal_coefficients = gpswf_reconstruct_tensor(
        recovered_qhat,
        target_basis,
        target_weights,
        modes,
        retained,
        image_matrix,
        grid_size,
        disk_mask,
    )
    reconstructed_tensor = tensor_matrix_from_coefficients(
        reconstructed_coefficients,
        kind,
    )

    _plot_tensor_components(
        truth_coefficients,
        reconstructed_coefficients,
        disk_mask,
        np.real,
        "real",
        (0, 4, 8),
        config.out_dir / "exp6_tensor_components_real.png",
    )
    _plot_tensor_components(
        truth_coefficients,
        reconstructed_coefficients,
        disk_mask,
        np.imag,
        "imaginary",
        (1, 3, 5),
        config.out_dir / "exp6_tensor_components_imag.png",
    )
    _plot_tensor_norm(
        truth_tensor,
        reconstructed_tensor,
        disk_mask,
        config.out_dir / "exp6_tensor_norm.png",
    )

    common_diagnostics: dict[str, Any] = {
        "experiment": 6,
        "k": k,
        "C": C,
        "K": K,
        "ell_max": ell_max,
        "n_modes_per_ell": n_modes_per_ell,
        "epsilon": epsilon,
        "N_cap": N_cap,
        "N_cap_theory": N_cap_theory,
        "N_cap_discrete": N_cap_discrete,
        "N_cap_hard": N_cap_hard,
        "noise_level": noise_level,
        "n_radial": n_radial,
        "n_angular_requested": n_angular,
        "n_per_axis": n_per_axis,
        "requested_measure_dirs": requested_measure_dirs,
        "actual_measure_dirs": int(data_info["n_measure_dirs"]),
        "candidate_count": int(data_info["candidate_count"]),
        "data_mode": "mock",
        "data_source": "full_vie",
        "tensor_kind": kind,
        "tensor_components": 9,
        "parameter_source": "Experiment 4 k=12 settings",
        **node_diagnostics(mock_distances),
        **mode_diagnostics(modes, retained, target_nodes_count=target_nodes.shape[0]),
        **projection_diagnostics(target_basis, target_weights, retained),
        **polarimetric_diagnostic_summary(dataset),
        **forward_solver_diagnostic_summary(dataset),
    }
    common_diagnostics.update({
        "C_mock_distance_mean": C * float(common_diagnostics["mock_distance_mean"]),
        "C_mock_distance_p95": C * float(common_diagnostics["mock_distance_p95"]),
        "C_mock_distance_max": C * float(common_diagnostics["mock_distance_max"]),
        "retained_multiplets": int(truncation_info["retained_multiplets"]),
        "partial_multiplets": int(truncation_info["partial_multiplets"]),
    })

    target_mask = disk_mask & (
        np.linalg.norm(truth_tensor, axis=(-2, -1)) > 1e-12
    )
    background_mask = disk_mask & ~target_mask
    diagnostic_rows: list[dict[str, Any]] = []
    for component, label in enumerate(COMPONENT_LABELS):
        truth_component = truth_coefficients[:, :, component]
        reconstructed_component = reconstructed_coefficients[:, :, component]
        absolute_reconstruction = np.abs(reconstructed_component)
        diagnostic_rows.append({
            **common_diagnostics,
            "metric_type": "component",
            "case_id": label,
            "component_index": component,
            "relative_l2_error": _relative_error(
                reconstructed_component[disk_mask],
                truth_component[disk_mask],
            ),
            "target_relative_l2_error": _relative_error(
                reconstructed_component[target_mask],
                truth_component[target_mask],
            ),
            "background_mean_abs": float(
                np.mean(absolute_reconstruction[background_mask])
            ),
            "background_p95_abs": float(
                np.percentile(absolute_reconstruction[background_mask], 95)
            ),
            "image_max_abs": float(np.max(absolute_reconstruction[disk_mask])),
            "qhat_relative_l2_error": _relative_error(
                recovered_qhat[:, component],
                truth_qhat[:, component],
            ),
            "qhat_norm": vector_norm(recovered_qhat[:, component]),
            **coefficient_diagnostics(modal_coefficients[:, component]),
        })

    block_masks = _block_masks(grid_points, grid_size, blocks)
    for block_index, (block, mask) in enumerate(zip(blocks, block_masks), start=1):
        mean_reconstruction = np.mean(reconstructed_tensor[mask], axis=0)
        truth_matrix = np.asarray(block.tensor, dtype=np.complex128)
        diagnostic_rows.append({
            **common_diagnostics,
            "metric_type": "inclusion",
            "case_id": f"block_{block_index}",
            "component_index": -1,
            "mean_tensor_relative_error": _relative_error(
                mean_reconstruction,
                truth_matrix,
            ),
            "truth_transpose_asymmetry": float(
                np.linalg.norm(truth_matrix - truth_matrix.T)
            ),
            "reconstructed_transpose_asymmetry": float(
                np.linalg.norm(mean_reconstruction - mean_reconstruction.T)
            ),
            "coercivity_min_eigenvalue": material_diagnostics[block_index - 1][
                "coercivity_min_eigenvalue"
            ],
            "passivity_min_eigenvalue": material_diagnostics[block_index - 1][
                "passivity_min_eigenvalue"
            ],
        })

    global_row: dict[str, Any] = {
        **common_diagnostics,
        "metric_type": "global",
        "case_id": "full_tensor",
        "component_index": -1,
        "tensor_relative_l2_error": _relative_error(
            reconstructed_tensor[disk_mask],
            truth_tensor[disk_mask],
        ),
        "tensor_target_relative_l2_error": _relative_error(
            reconstructed_tensor[target_mask],
            truth_tensor[target_mask],
        ),
        "qhat_tensor_relative_l2_error": _relative_error(
            recovered_qhat,
            truth_qhat,
        ),
        "tensor_background_mean_frobenius": float(np.mean(
            np.linalg.norm(reconstructed_tensor[background_mask], axis=(-2, -1))
        )),
        "tensor_background_p95_frobenius": float(np.percentile(
            np.linalg.norm(reconstructed_tensor[background_mask], axis=(-2, -1)),
            95,
        )),
    }
    for row, column in ((0, 1), (0, 2), (1, 2)):
        truth_skew = truth_tensor[..., row, column] - truth_tensor[..., column, row]
        reconstructed_skew = (
            reconstructed_tensor[..., row, column]
            - reconstructed_tensor[..., column, row]
        )
        global_row[f"Q{row + 1}{column + 1}_Q{column + 1}{row + 1}_skew_relative_error"] = (
            _relative_error(
                reconstructed_skew[disk_mask],
                truth_skew[disk_mask],
            )
        )
    diagnostic_rows.append(global_row)
    write_diagnostics_csv(
        diagnostic_rows,
        config.out_dir / "exp6_tensor_diagnostics.csv",
    )

    np.savez_compressed(
        config.out_dir / "exp6_tensor_diagnostics_detail.npz",
        component_labels=np.asarray(COMPONENT_LABELS),
        block_centers=np.asarray([block.center for block in blocks], dtype=float),
        block_half_widths=np.asarray([block.half_width for block in blocks], dtype=float),
        block_tensors=np.asarray([block.tensor for block in blocks]),
        target_nodes=target_nodes,
        target_weights=target_weights,
        measurement_nodes=measurement_nodes,
        mock_distances=mock_distances,
        matched_nodes=np.asarray(data_info["matched_nodes"]),
        incident_dirs=dataset.incident_dirs,
        obs_dirs=dataset.obs_dirs,
        clean_farfield=dataset.farfield_data,
        noisy_farfield=noisy_farfield,
        recovered_qhat=recovered_qhat,
        truth_qhat=truth_qhat,
        truth_coefficients=truth_coefficients,
        reconstructed_coefficients=reconstructed_coefficients,
        truth_tensor=truth_tensor,
        reconstructed_tensor=reconstructed_tensor,
        modal_coefficients=modal_coefficients,
        retained=retained,
        alpha=np.asarray([mode.alpha for mode in modes]),
        polarimetric_ranks=np.asarray(dataset.metadata["polarimetric_ranks"]),
        polarimetric_sigma_min=np.asarray(dataset.metadata["polarimetric_sigma_min"]),
        polarimetric_condition_numbers=np.asarray(
            dataset.metadata["polarimetric_condition_numbers"]
        ),
    )

    metadata = {
        "experiment": 6,
        "description": "piecewise non-reciprocal full-tensor reconstruction",
        "quick": bool(config.quick),
        "seed": int(config.seed),
        "tensor_kind": kind,
        "coefficient_order": list(COMPONENT_LABELS),
        "displayed_components": {
            "real": ["Q11", "Q22", "Q33"],
            "imaginary": ["Q12", "Q21", "Q23"],
            "color_scale": "one scale per component, shared by truth and GPSWF",
        },
        "fourier_convention": "exp(-i C p.x)",
        "k": k,
        "C": C,
        "noise_level": noise_level,
        "polarimetric_J": polarimetric_J,
        "parameters": {
            "K": K,
            "ell_max": ell_max,
            "n_modes_per_ell": n_modes_per_ell,
            "n_radial": n_radial,
            "n_angular_requested": n_angular,
            "grid_size": grid_size,
            "n_per_axis": n_per_axis,
            "requested_measure_dirs": requested_measure_dirs,
            "parameter_source": "Experiment 4 k=12 settings",
        },
        "truncation": {
            "epsilon": epsilon,
            "N_cap": N_cap,
            "N_cap_theory": N_cap_theory,
            "N_cap_discrete": N_cap_discrete,
            "N_cap_hard": N_cap_hard,
            **{key: value for key, value in truncation_info.items()},
        },
        "polarimetric": {
            **polarimetric_diagnostic_summary(dataset),
            "candidate_count": int(data_info["candidate_count"]),
            "actual_measure_dirs": int(data_info["n_measure_dirs"]),
        },
        "mock_distance": node_diagnostics(mock_distances),
        "forward": forward_solver_diagnostic_summary(dataset),
        "blocks": [
            {
                "center": list(block.center),
                "half_width": list(block.half_width),
                "tensor": _complex_matrix_json(block.tensor),
                "material_diagnostics": material_diagnostics[index],
            }
            for index, block in enumerate(blocks)
        ],
        "outputs": [
            "exp6_tensor_components_real.png",
            "exp6_tensor_components_imag.png",
            "exp6_tensor_norm.png",
            "exp6_tensor_diagnostics.csv",
            "exp6_tensor_diagnostics_detail.npz",
            "metadata.json",
        ],
    }
    with (config.out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(
        "Experiment 6: "
        f"rank_min={int(dataset.metadata['polarimetric_rank_min'])}, "
        f"retained={int(np.sum(retained))}, "
        f"tensor_relerr={global_row['tensor_relative_l2_error']:.3e}"
    )
    return {
        "diagnostics": diagnostic_rows,
        "metadata": metadata,
        "truth_tensor": truth_tensor,
        "reconstructed_tensor": reconstructed_tensor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 6: piecewise full-tensor reconstruction",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/exp6")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        ExperimentConfig(
            out_dir=Path(args.out_dir),
            seed=args.seed,
            quick=args.quick,
        )
    )


if __name__ == "__main__":
    main()
