#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared pipeline for Maxwell basis-comparison experiments.

The forward generator remains separate from the inverse stage:

    Q(x) -> far-field dataset -> polarimetric Qhat -> four reconstructions.

Experiment scripts select the forward data source and row layout.  This module
keeps the GPSWF, cube Fourier, ball Bessel, DSM, plotting, and diagnostics
settings identical across those rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.config import ExperimentConfig
from common.diagnostics import (
    collect_reconstruction_diagnostics,
    plot_diagnostic_curves,
    save_diagnostics_npz,
    write_diagnostics_csv,
)
from common.direct_sampling import direct_sampling_component_indicator
from common.fourier import reconstruct_fourier_cube_from_data
from common.gpswf import (
    collect_alpha_pairs_cached,
    eval_bandlimited_radial_extension,
    gpswf_spatial_concentration,
    modal_matrix,
    quadrature_modal_coefficients,
    solve_ball_gpswf,
)
from common.phantom import (
    Mode,
    tensor_truth_image_2d,
    three_admissible_full_tensor_blocks,
)
from common.quadrature import (
    ball_quadrature_nodes,
    generate_polarimetric_data_nodes,
)
from common.reconstruction import select_complete_gpswf_multiplets
from common.spherical_bessel import reconstruct_ball_bessel_from_data
from common.utils import make_table, vector_norm
from forward.datasets import (
    FarfieldDataset,
    analytic_tensor_block_born_farfield_dataset,
    discrete_vie_born_farfield_dataset,
    farfield_dataset_to_qhat,
    full_vie_farfield_dataset,
    polarimetric_diagnostic_summary,
)

ForwardSource = Literal["analytic_born", "vie_born", "full_vie"]


@dataclass(frozen=True)
class BasisComparisonRow:
    """One plotted row and its forward data source."""

    k: float
    label: str
    data_source: ForwardSource


@dataclass(frozen=True)
class _MeasurementSetup:
    target_nodes: np.ndarray
    target_weights: np.ndarray
    measurement_nodes: np.ndarray
    incident_dirs: np.ndarray
    obs_dirs: np.ndarray
    mock_distances: np.ndarray
    data_info: dict[str, Any]
    params: dict[str, float | int]
    n_per_axis: int
    requested_measure_dirs: int


@dataclass(frozen=True)
class _ForwardRow:
    spec: BasisComparisonRow
    setup: _MeasurementSetup
    dataset: FarfieldDataset
    recovered: np.ndarray
    component_data: np.ndarray
    polarimetric_diagnostics: dict[str, float | int]
    farfield_error_vs_analytic: float
    qhat_error_vs_analytic: float
    component_qhat_error_vs_analytic: float


@dataclass(frozen=True)
class _BasisContext:
    modes: list[Mode]
    retained: np.ndarray
    truncation_info: dict[str, int]
    target_basis: np.ndarray
    image_matrix: np.ndarray
    grid_points: np.ndarray
    truth: np.ndarray
    disk_mask: np.ndarray
    square_mask: np.ndarray
    vmin: float
    vmax: float
    N_cap: int
    N_cap_theory: int
    N_cap_discrete: int
    N_cap_hard: int
    fourier_mode_cap: int
    bessel_mode_cap: int
    gpswf_retained_count: int


def basis_row_params(k: float) -> dict[str, float | int]:
    """Return the formal GPSWF and ball-quadrature parameters for ``k``."""
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
    if k <= 10:
        return {"ell_max": 8, "n_modes": 5, "K": 36, "n_radial": 10, "n_angular": 110, "C": C}
    if k <= 12:
        return {"ell_max": 9, "n_modes": 5, "K": 40, "n_radial": 10, "n_angular": 170, "C": C}
    if k <= 16:
        return {"ell_max": 12, "n_modes": 7, "K": 48, "n_radial": 12, "n_angular": 230, "C": C}
    if k <= 20:
        return {"ell_max": 14, "n_modes": 7, "K": 54, "n_radial": 14, "n_angular": 302, "C": C}
    return {"ell_max": 16, "n_modes": 7, "K": 60, "n_radial": 14, "n_angular": 302, "C": C}


def basis_row_params_quick(k: float) -> dict[str, float | int]:
    """Return reduced parameters used only by local smoke tests."""
    C = 2.0 * k
    if k <= 4:
        return {"ell_max": 2, "n_modes": 2, "K": 10, "n_radial": 3, "n_angular": 26, "C": C}
    if k <= 6:
        return {"ell_max": 4, "n_modes": 2, "K": 14, "n_radial": 5, "n_angular": 38, "C": C}
    if k <= 8:
        return {"ell_max": 5, "n_modes": 3, "K": 16, "n_radial": 5, "n_angular": 50, "C": C}
    if k <= 9:
        return {"ell_max": 6, "n_modes": 3, "K": 18, "n_radial": 5, "n_angular": 74, "C": C}
    if k <= 10:
        return {"ell_max": 7, "n_modes": 3, "K": 20, "n_radial": 6, "n_angular": 74, "C": C}
    if k <= 12:
        return {"ell_max": 8, "n_modes": 3, "K": 22, "n_radial": 6, "n_angular": 86, "C": C}
    if k <= 16:
        return {"ell_max": 10, "n_modes": 4, "K": 28, "n_radial": 8, "n_angular": 110, "C": C}
    if k <= 20:
        return {"ell_max": 12, "n_modes": 5, "K": 32, "n_radial": 10, "n_angular": 146, "C": C}
    return {"ell_max": 14, "n_modes": 5, "K": 36, "n_radial": 10, "n_angular": 170, "C": C}


def basis_n_per_axis(
    k: float,
    quick: bool,
    experiment_number: int | None = None,
) -> int:
    """Return the voxel-grid side count used by VIE-based data sources."""
    if quick:
        return 7
    if experiment_number == 4:
        formal_values = {8: 13, 12: 19, 15: 23}
        key = int(k)
        if float(key) == float(k) and key in formal_values:
            return formal_values[key]
        return max(11, int(math.ceil((23.0 / 15.0) * float(k))))
    return max(11, min(19, int(k * 1.2)))


def basis_measure_dirs(
    k: float,
    quick: bool,
    experiment_number: int | None = None,
) -> int:
    """Return finite measurement directions anchored at k=15 -> 974."""
    if quick:
        return 110
    if experiment_number == 4:
        formal_values = {8: 434, 12: 770, 15: 974}
        key = int(k)
        if float(key) == float(k) and key in formal_values:
            return formal_values[key]
    return 974


def _settings(quick: bool) -> tuple[int, int, int]:
    if quick:
        return 51, 100, 80
    return 161, 160, 120


def _relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(vector_norm(reference), 1e-14)
    return vector_norm(values - reference) / denominator


def _build_dataset(
    source: ForwardSource,
    *,
    k: float,
    setup: _MeasurementSetup,
    polarimetric_J: int,
) -> FarfieldDataset:
    blocks = three_admissible_full_tensor_blocks()
    common_kwargs = {
        "kind": "full",
        "k": float(k),
        "incident_dirs": setup.incident_dirs,
        "obs_dirs": setup.obs_dirs,
    }
    if source == "analytic_born":
        return analytic_tensor_block_born_farfield_dataset(
            blocks,
            setup.measurement_nodes,
            **common_kwargs,
        )
    if source == "vie_born":
        return discrete_vie_born_farfield_dataset(
            "piecewise_full_tensor_blocks",
            setup.measurement_nodes,
            R=1.0,
            n_per_axis=setup.n_per_axis,
            n_geometries=polarimetric_J,
            **common_kwargs,
        )
    if source == "full_vie":
        return full_vie_farfield_dataset(
            "piecewise_full_tensor_blocks",
            setup.measurement_nodes,
            R=1.0,
            n_per_axis=setup.n_per_axis,
            n_geometries=polarimetric_J,
            **common_kwargs,
        )
    raise ValueError(f"Unknown forward source: {source!r}")


def _measurement_setup(
    k: float,
    *,
    quick: bool,
    experiment_number: int,
    polarimetric_J: int,
) -> _MeasurementSetup:
    params = basis_row_params_quick(k) if quick else basis_row_params(k)
    requested_measure_dirs = basis_measure_dirs(
        k,
        quick,
        experiment_number=experiment_number,
    )
    target_nodes, target_weights, _ = ball_quadrature_nodes(
        int(params["n_radial"]),
        int(params["n_angular"]),
    )
    physical_nodes, incident_dirs, obs_dirs, distances, data_info = (
        generate_polarimetric_data_nodes(
            -target_nodes,
            requested_measure_dirs,
            polarimetric_J=polarimetric_J,
            tensor_kind="full",
        )
    )
    return _MeasurementSetup(
        target_nodes=target_nodes,
        target_weights=target_weights,
        measurement_nodes=-physical_nodes,
        incident_dirs=incident_dirs,
        obs_dirs=obs_dirs,
        mock_distances=distances,
        data_info=data_info,
        params=params,
        n_per_axis=basis_n_per_axis(
            k,
            quick,
            experiment_number=experiment_number,
        ),
        requested_measure_dirs=requested_measure_dirs,
    )


def _forward_metadata(dataset: FarfieldDataset) -> dict[str, float | int]:
    metadata = dataset.metadata
    keys = (
        "vie_voxel_nodes",
        "vie_unknowns",
        "vie_unique_incident_directions",
        "vie_rhs_count",
        "vie_residual_sample_count",
        "vie_linear_residual_sample_max",
    )
    result: dict[str, float | int] = {}
    for key in keys:
        value = metadata.get(key, math.nan)
        result[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return result


def _prepare_forward_rows(
    config: ExperimentConfig,
    row_specs: list[BasisComparisonRow],
    *,
    experiment_number: int,
    noise_level: float,
    polarimetric_J: int,
    shared_noise_for_equal_k: bool,
) -> list[_ForwardRow]:
    setup_cache: dict[float, _MeasurementSetup] = {}
    raw_rows: list[tuple[BasisComparisonRow, _MeasurementSetup, FarfieldDataset]] = []
    for spec in row_specs:
        setup = setup_cache.get(float(spec.k))
        if setup is None:
            setup = _measurement_setup(
                spec.k,
                quick=config.quick,
                experiment_number=experiment_number,
                polarimetric_J=polarimetric_J,
            )
            setup_cache[float(spec.k)] = setup
        raw_rows.append((spec, setup, _build_dataset(
            spec.data_source,
            k=spec.k,
            setup=setup,
            polarimetric_J=polarimetric_J,
        )))

    rng = np.random.default_rng(config.seed + experiment_number * 100)
    standard_noise_by_k: dict[float, np.ndarray] = {}
    recovered_rows: list[tuple[
        BasisComparisonRow,
        _MeasurementSetup,
        FarfieldDataset,
        np.ndarray,
    ]] = []
    for spec, setup, dataset in raw_rows:
        standard_noise = None
        if shared_noise_for_equal_k:
            standard_noise = standard_noise_by_k.get(float(spec.k))
            if standard_noise is None:
                shape = dataset.farfield_data.shape
                standard_noise = (
                    rng.normal(size=shape) + 1j * rng.normal(size=shape)
                )
                standard_noise_by_k[float(spec.k)] = standard_noise
        recovered = farfield_dataset_to_qhat(
            dataset,
            kind="full",
            noise_level=noise_level,
            rng=None if standard_noise is not None else rng,
            standard_noise=standard_noise,
        )
        recovered_rows.append((spec, setup, dataset, recovered))

    analytic_by_k: dict[float, tuple[FarfieldDataset, np.ndarray]] = {}
    for spec, _, dataset, recovered in recovered_rows:
        if spec.data_source == "analytic_born":
            analytic_by_k[float(spec.k)] = (dataset, recovered)

    result: list[_ForwardRow] = []
    for spec, setup, dataset, recovered in recovered_rows:
        reference = analytic_by_k.get(float(spec.k))
        if reference is None:
            farfield_error = math.nan
            qhat_error = math.nan
            component_error = math.nan
        else:
            reference_dataset, reference_qhat = reference
            farfield_error = _relative_error(
                dataset.farfield_data,
                reference_dataset.farfield_data,
            )
            qhat_error = _relative_error(recovered, reference_qhat)
            component_error = _relative_error(recovered[:, 0], reference_qhat[:, 0])
        result.append(_ForwardRow(
            spec=spec,
            setup=setup,
            dataset=dataset,
            recovered=recovered,
            component_data=recovered[:, 0],
            polarimetric_diagnostics=polarimetric_diagnostic_summary(dataset),
            farfield_error_vs_analytic=farfield_error,
            qhat_error_vs_analytic=qhat_error,
            component_qhat_error_vs_analytic=component_error,
        ))
    return result


def _prepare_basis_context(
    config: ExperimentConfig,
    setup: _MeasurementSetup,
    *,
    grid_size: int,
    quad_order: int,
    r_eval_count: int,
    epsilon: float,
    fourier_mode_fraction: float,
    bessel_mode_fraction: float,
    basis_mode_min: int,
) -> _BasisContext:
    params = setup.params
    C = float(params["C"])
    ell_max = int(params["ell_max"])
    n_modes_per_ell = int(params["n_modes"])
    K_val = int(params["K"])

    alpha_table = collect_alpha_pairs_cached(
        C,
        K_val,
        ell_max,
        n_modes_per_ell,
        quad_order=quad_order,
        r_eval_count=r_eval_count,
        cache_dir=config.out_dir / "alpha_cache",
    )
    alpha_lookup = {
        (int(row["ell"]), int(row["n"])): complex(
            float(row["alpha_real"]),
            float(row["alpha_imag"]),
        )
        for _, row in alpha_table.iterrows()
    }
    modes: list[Mode] = []
    chi_lookup: dict[tuple[int, int], float] = {}
    for ell in range(ell_max + 1):
        chi, beta = solve_ball_gpswf(
            C,
            ell,
            K_val,
            n_modes=n_modes_per_ell,
        )
        for n in range(beta.shape[1]):
            alpha = alpha_lookup[(ell, n)]
            chi_lookup[(ell, n)] = float(chi[n])
            for m in range(-ell, ell + 1):
                modes.append(Mode(
                    ell=ell,
                    n=n,
                    m=m,
                    alpha=alpha,
                    beta=beta[:, n],
                ))

    alpha_abs = np.asarray([abs(mode.alpha) for mode in modes], dtype=float)
    eligible = alpha_abs > epsilon * float(np.max(alpha_abs))
    N_cap_theory = int(C * C / 2)
    N_cap_discrete = setup.target_nodes.shape[0] // 6
    N_cap_hard = 512
    N_cap = min(N_cap_theory, N_cap_discrete, N_cap_hard)
    retained, truncation_info = select_complete_gpswf_multiplets(
        modes,
        chi_lookup,
        N_cap,
        eligible=eligible,
    )
    gpswf_count = int(np.sum(retained))
    fourier_mode_cap = min(
        N_cap_hard,
        max(int(basis_mode_min), int(fourier_mode_fraction * gpswf_count)),
    )
    bessel_mode_cap = min(
        N_cap_hard,
        max(int(basis_mode_min), int(bessel_mode_fraction * gpswf_count)),
    )

    blocks = three_admissible_full_tensor_blocks()
    truth, _, disk_mask = tensor_truth_image_2d(
        grid_size,
        blocks,
        kind="full",
        component_index=0,
        display="component",
    )
    xs = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(xs, xs)
    grid_points = np.column_stack([
        X.reshape(-1),
        Y.reshape(-1),
        np.zeros(grid_size * grid_size),
    ])
    return _BasisContext(
        modes=modes,
        retained=retained,
        truncation_info=truncation_info,
        target_basis=modal_matrix(setup.target_nodes, modes, fourier_side=True),
        image_matrix=modal_matrix(grid_points, modes, fourier_side=False),
        grid_points=grid_points,
        truth=truth,
        disk_mask=disk_mask,
        square_mask=np.ones_like(disk_mask, dtype=bool),
        vmin=float(np.nanmin(np.real(truth))),
        vmax=float(np.nanmax(np.real(truth))),
        N_cap=N_cap,
        N_cap_theory=N_cap_theory,
        N_cap_discrete=N_cap_discrete,
        N_cap_hard=N_cap_hard,
        fourier_mode_cap=fourier_mode_cap,
        bessel_mode_cap=bessel_mode_cap,
        gpswf_retained_count=gpswf_count,
    )


def _baseline_diagnostics(
    *,
    case: dict[str, Any],
    image: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(case)
    abs_image = np.abs(image)
    target_mask = valid_mask & (np.abs(truth) > 1e-12)
    background_mask = valid_mask & ~target_mask

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_image[mask])) if np.any(mask) else math.nan

    def p95(mask: np.ndarray) -> float:
        return float(np.percentile(abs_image[mask], 95)) if np.any(mask) else math.nan

    row.update({
        "retained_modes": int(row.get("basis_modes", row.get("retained_N", 0))),
        "total_modes": int(row.get("basis_modes", row.get("total_modes", 0))),
        "gram_offdiag_ratio": math.nan,
        "gram_cond": math.nan,
        "image_min": float(np.min(np.real(image))),
        "image_max": float(np.max(np.real(image))),
        "image_max_abs": float(np.max(abs_image)),
        "image_l2": vector_norm(image),
        "background_mean_abs": mean(background_mask),
        "background_p95_abs": p95(background_mask),
        "target_mean_abs": mean(target_mask),
        "target_p95_abs": p95(target_mask),
    })
    return row


def _case_common(
    forward_row: _ForwardRow,
    *,
    experiment_number: int,
    row_index: int,
    column_index: int,
    method: str,
    noise_level: float,
    polarimetric_J: int,
) -> dict[str, Any]:
    setup = forward_row.setup
    params = setup.params
    component_data = forward_row.component_data
    return {
        "experiment": int(experiment_number),
        "method": method,
        "row": int(row_index),
        "column": int(column_index),
        "k": float(forward_row.spec.k),
        "C": float(params["C"]),
        "noise_level": float(noise_level),
        "n_radial": int(params["n_radial"]),
        "n_angular_requested": int(params["n_angular"]),
        "n_per_axis": int(setup.n_per_axis),
        "n_geometries": int(polarimetric_J),
        "requested_measure_dirs": int(setup.requested_measure_dirs),
        "candidate_count": int(setup.data_info["candidate_count"]),
        "data_mode": "mock",
        "data_source": forward_row.spec.data_source,
        "shape": "piecewise_full_tensor_blocks_gap_0.20",
        "farfield_error_vs_analytic": float(forward_row.farfield_error_vs_analytic),
        "qhat_error_vs_analytic": float(forward_row.qhat_error_vs_analytic),
        "component_qhat_error_vs_analytic": float(
            forward_row.component_qhat_error_vs_analytic
        ),
        "data_norm": vector_norm(component_data),
        "data_max_abs": float(np.max(np.abs(component_data))),
        **forward_row.polarimetric_diagnostics,
        **_forward_metadata(forward_row.dataset),
    }


def _imshow(ax: Any, image: np.ndarray, title: str, vmin: float, vmax: float):
    result = ax.imshow(
        image,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return result


def _multiplet_concentration_records(
    context: _BasisContext,
    C: float,
) -> list[dict[str, Any]]:
    """Collect one spatial-concentration record per complete ``(ell, n)`` block."""
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    representatives: dict[tuple[int, int], Mode] = {}
    for mode_index, mode in enumerate(context.modes):
        key = (int(mode.ell), int(mode.n))
        grouped_indices.setdefault(key, []).append(mode_index)
        representatives.setdefault(key, mode)

    records: list[dict[str, Any]] = []
    for key, indices in grouped_indices.items():
        mode = representatives[key]
        retained_values = np.asarray(context.retained[indices], dtype=bool)
        if np.any(retained_values) and not np.all(retained_values):
            raise RuntimeError(f"partial GPSWF multiplet encountered for {key}")
        records.append({
            "ell": key[0],
            "n": key[1],
            "mode": mode,
            "degeneracy": 2 * key[0] + 1,
            "retained": bool(np.all(retained_values)),
            "concentration": gpswf_spatial_concentration(C, mode.alpha),
        })
    return records


def _plot_gpswf_spatial_concentration(
    forward_rows: list[_ForwardRow],
    context_cache: dict[float, _BasisContext],
    output_path: Path,
) -> None:
    """Plot retained-mode density and exact modal energy fractions in ``B_R``."""
    n_rows = len(forward_rows)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9.0, 3.2 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[None, :]

    plane_half_width = 1.8
    radial_nodes = np.linspace(0.0, math.sqrt(2.0) * plane_half_width, 481)
    plane_axis = np.linspace(-plane_half_width, plane_half_width, 321)
    plane_x, plane_y = np.meshgrid(plane_axis, plane_axis)
    plane_radius = np.sqrt(plane_x * plane_x + plane_y * plane_y)
    density_images: list[np.ndarray] = []
    row_records: list[list[dict[str, Any]]] = []

    for forward_row in forward_rows:
        k = float(forward_row.spec.k)
        C = float(forward_row.setup.params["C"])
        context = context_cache[k]
        records = _multiplet_concentration_records(context, C)
        retained_records = [record for record in records if record["retained"]]
        if not retained_records:
            raise RuntimeError(f"no retained GPSWF modes for k={k:g}")

        radial_density = np.zeros_like(radial_nodes)
        retained_mode_count = 0
        for record in retained_records:
            mode = record["mode"]
            concentration = max(float(record["concentration"]), 0.0)
            radial_extension = eval_bandlimited_radial_extension(
                radial_nodes,
                C,
                int(mode.ell),
                mode.beta,
                mode.alpha,
            )
            # The B_1-normalised extension has global norm squared 1/eta.
            # Multiplying by sqrt(eta) gives every global mode unit energy.
            radial_global_unit = math.sqrt(concentration) * radial_extension
            degeneracy = int(record["degeneracy"])
            radial_density += (
                degeneracy
                * np.abs(radial_global_unit) ** 2
                / (4.0 * math.pi)
            )
            retained_mode_count += degeneracy

        radial_density /= max(retained_mode_count, 1)
        plane_density = np.interp(
            plane_radius.reshape(-1),
            radial_nodes,
            radial_density,
        ).reshape(plane_radius.shape)
        peak = max(float(np.max(plane_density)), 1e-300)
        density_images.append(np.log10(np.maximum(plane_density / peak, 1e-6)))
        row_records.append(records)

    image_artist = None
    for row_index, (forward_row, density_image, records) in enumerate(
        zip(forward_rows, density_images, row_records)
    ):
        k = float(forward_row.spec.k)
        image_artist = axes[row_index, 0].imshow(
            density_image,
            origin="lower",
            extent=(-1.8, 1.8, -1.8, 1.8),
            cmap="magma",
            vmin=-6.0,
            vmax=0.0,
            interpolation="bilinear",
        )
        axes[row_index, 0].add_patch(
            plt.Circle((0.0, 0.0), 1.0, fill=False, color="cyan", linewidth=1.4)
        )
        axes[row_index, 0].set_aspect("equal")
        axes[row_index, 0].set_xlabel(r"$x_1/R$")
        axes[row_index, 0].set_ylabel(r"$x_2/R$")

        retained = [record for record in records if record["retained"]]
        retained_weight = sum(int(record["degeneracy"]) for record in retained)
        mean_concentration = sum(
            int(record["degeneracy"]) * float(record["concentration"])
            for record in retained
        ) / max(retained_weight, 1)
        min_concentration = min(float(record["concentration"]) for record in retained)
        axes[row_index, 0].set_title(
            rf"$k={k:g}$ retained density; "
            rf"mean $E(B_R)={mean_concentration:.3f}$"
        )

        sorted_records = sorted(
            records,
            key=lambda record: float(record["concentration"]),
            reverse=True,
        )
        ranks = np.arange(1, len(sorted_records) + 1)
        concentration = np.asarray(
            [float(record["concentration"]) for record in sorted_records]
        )
        retained_mask = np.asarray(
            [bool(record["retained"]) for record in sorted_records]
        )
        axes[row_index, 1].plot(ranks, concentration, color="0.65", linewidth=1.0)
        axes[row_index, 1].scatter(
            ranks[retained_mask],
            concentration[retained_mask],
            s=25,
            color="#0072B2",
            label="retained",
            zorder=3,
        )
        if np.any(~retained_mask):
            axes[row_index, 1].scatter(
                ranks[~retained_mask],
                concentration[~retained_mask],
                s=28,
                marker="x",
                color="#D55E00",
                label="discarded",
                zorder=3,
            )
        axes[row_index, 1].axhline(0.9, color="0.35", linestyle="--", linewidth=0.9)
        axes[row_index, 1].set_xlim(0.5, len(sorted_records) + 0.5)
        axes[row_index, 1].set_ylim(0.0, 1.05)
        axes[row_index, 1].set_xlabel("multiplet rank")
        axes[row_index, 1].set_ylabel(r"$E_{\ell n}(B_R)$")
        axes[row_index, 1].set_title(
            rf"spatial concentration; retained min $={min_concentration:.3f}$"
        )
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend(fontsize=7, loc="lower left")

    if image_artist is not None:
        fig.colorbar(
            image_artist,
            ax=axes[:, 0],
            fraction=0.025,
            pad=0.02,
            label=r"$\log_{10}(\rho/\rho_{\max})$",
        )
    fig.suptitle(
        r"GPSWF spatial energy concentration inside $B_R$ (cyan boundary)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _select_representative_multiplets(
    records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Choose distinct retained multiplets with complementary structures."""
    retained = [record for record in records if record["retained"]]
    if not retained:
        raise RuntimeError("no retained GPSWF multiplets are available")

    proposals: list[tuple[str, dict[str, Any]]] = []
    proposals.append(("fundamental", min(
        retained, key=lambda record: (int(record["ell"]), int(record["n"]))
    )))
    retained_by_key = {
        (int(record["ell"]), int(record["n"])): record
        for record in retained
    }
    for nearby_key in ((0, 1), (1, 0)):
        nearby_record = retained_by_key.get(nearby_key)
        if nearby_record is not None:
            proposals.append(("near fundamental", nearby_record))
    radial = [record for record in retained if int(record["ell"]) == 0]
    if radial:
        proposals.append(("radial high-order", max(
            radial, key=lambda record: int(record["n"])
        )))
    angular = [record for record in retained if int(record["n"]) == 0]
    if angular:
        proposals.append(("angular high-order", max(
            angular, key=lambda record: int(record["ell"])
        )))
    discarded = [record for record in records if not record["retained"]]
    if discarded:
        proposals.append(("weak / discarded", min(
            discarded, key=lambda record: float(record["concentration"])
        )))
    else:
        proposals.append(("least concentrated retained", min(
            retained, key=lambda record: float(record["concentration"])
        )))

    selected: list[tuple[str, dict[str, Any]]] = []
    selected_keys: set[tuple[int, int]] = set()
    for label, record in proposals:
        key = (int(record["ell"]), int(record["n"]))
        if key not in selected_keys:
            selected.append((label, record))
            selected_keys.add(key)

    # Small quick-mode bases can make proposals coincide or omit a requested
    # neighbour. Fill remaining slots with simple distinct retained modes.
    for record in sorted(
        retained,
        key=lambda item: (int(item["ell"]) + 2 * int(item["n"]), int(item["ell"])),
    ):
        key = (int(record["ell"]), int(record["n"]))
        if key in selected_keys:
            continue
        selected.append(("additional retained", record))
        selected_keys.add(key)
        if len(selected) == 6:
            break
    return selected[:6]


def _plot_gpswf_representative_modes(
    context: _BasisContext,
    C: float,
    k: float,
    output_path: Path,
) -> None:
    """Plot unrescaled spherical maxima of representative eigenmodes."""
    records = _multiplet_concentration_records(context, C)
    selected = _select_representative_multiplets(records)
    radial_nodes = np.linspace(0.0, 2.0, 801)
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    ax.axvspan(0.0, 1.0, color="#56B4E9", alpha=0.10, label=r"inside $B_R$")
    ax.axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"boundary $r=R$",
    )
    for curve_index, (role, record) in enumerate(selected):
        mode = record["mode"]
        ell = int(record["ell"])
        n = int(record["n"])
        m = ell if ell > 0 else 0
        radial_extension = eval_bandlimited_radial_extension(
            radial_nodes,
            C,
            ell,
            mode.beta,
            mode.alpha,
        )
        # The selected representative uses m=ell.  For the L2(S^2)-normalised
        # sectoral spherical harmonic, its exact maximum is attained at the
        # equator.  Keep this factor because the curves now share the original
        # L2(B_R) modal scale instead of being divided by separate peaks.
        log_angular_maximum_squared = (
            math.log(2 * ell + 1)
            - math.log(4.0 * math.pi)
            + math.lgamma(2 * ell + 1)
            - 2.0 * ell * math.log(2.0)
            - 2.0 * math.lgamma(ell + 1)
        )
        angular_maximum = math.exp(0.5 * log_angular_maximum_squared)
        spherical_maximum = angular_maximum * np.abs(radial_extension)
        ax.plot(
            radial_nodes,
            spherical_maximum,
            linewidth=1.35,
            color=colors[curve_index % len(colors)],
            label=rf"$(\ell,n,m)=({ell},{n},{m})$",
        )
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"normalised radius $r/R$")
    ax.set_ylabel(r"$\max_{|x|=r}|\widetilde\psi_{\ell n m}(x)|$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_basis_comparison(
    config: ExperimentConfig,
    *,
    experiment_number: int,
    row_specs: list[BasisComparisonRow],
    output_stem: str,
    figure_title: str,
    noise_level: float = 0.2,
    shared_noise_for_equal_k: bool = False,
    include_dsm: bool = True,
    gpswf_spatial_concentration_output: str | None = None,
    gpswf_representative_modes_output: str | None = None,
) -> Any:
    """Run one basis comparison with explicit forward-source rows."""
    if not row_specs:
        raise ValueError("row_specs must not be empty")

    grid_size, quad_order, r_eval_count = _settings(config.quick)
    polarimetric_J = 6
    epsilon = 0.2
    half_side = 1.0
    bandwidth_factor = 2.0
    basis_lstsq_rcond = 1e-8
    fourier_mode_fraction = 1.2
    bessel_mode_fraction = 1.2
    basis_mode_min = 12

    forward_rows = _prepare_forward_rows(
        config,
        row_specs,
        experiment_number=experiment_number,
        noise_level=noise_level,
        polarimetric_J=polarimetric_J,
        shared_noise_for_equal_k=shared_noise_for_equal_k,
    )

    context_cache: dict[float, _BasisContext] = {}
    n_rows = len(forward_rows)
    n_cols = 4 if include_dsm else 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.1 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[None, :]
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_panels: list[tuple[np.ndarray, str, np.ndarray]] = []

    for row_index, forward_row in enumerate(forward_rows):
        spec = forward_row.spec
        setup = forward_row.setup
        params = setup.params
        C = float(params["C"])
        context = context_cache.get(float(spec.k))
        if context is None:
            context = _prepare_basis_context(
                config,
                setup,
                grid_size=grid_size,
                quad_order=quad_order,
                r_eval_count=r_eval_count,
                epsilon=epsilon,
                fourier_mode_fraction=fourier_mode_fraction,
                bessel_mode_fraction=bessel_mode_fraction,
                basis_mode_min=basis_mode_min,
            )
            context_cache[float(spec.k)] = context
        component_data = forward_row.component_data

        gpswf_coeffs = quadrature_modal_coefficients(
            component_data,
            context.target_basis,
            setup.target_weights,
            context.modes,
            context.retained,
        )
        gpswf_rec = (context.image_matrix @ gpswf_coeffs).reshape(
            grid_size,
            grid_size,
        )
        gpswf_rec[~context.disk_mask] = 0.0
        gpswf_image = np.real(gpswf_rec)
        gpswf_title = f"GPSWF\nN={context.gpswf_retained_count}"
        reconstruction_panels.append((gpswf_image, gpswf_title, context.disk_mask))
        _imshow(
            axes[row_index, 0],
            gpswf_image,
            gpswf_title,
            context.vmin,
            context.vmax,
        )
        gpswf_case = _case_common(
            forward_row,
            experiment_number=experiment_number,
            row_index=row_index,
            column_index=0,
            method="gpswf",
            noise_level=noise_level,
            polarimetric_J=polarimetric_J,
        )
        gpswf_case.update({
            "case_id": _case_id(experiment_number, spec, "gpswf"),
            "K": int(params["K"]),
            "ell_max": int(params["ell_max"]),
            "n_modes_per_ell": int(params["n_modes"]),
            "epsilon": float(epsilon),
            "N_cap": int(context.N_cap),
            "N_cap_theory": int(context.N_cap_theory),
            "N_cap_discrete": int(context.N_cap_discrete),
            "N_cap_hard": int(context.N_cap_hard),
            "retained_N": int(context.gpswf_retained_count),
            "retained_multiplets": int(context.truncation_info["retained_multiplets"]),
            "partial_multiplets": int(context.truncation_info["partial_multiplets"]),
            "support": "unit_ball",
        })
        diagnostic_rows.append(collect_reconstruction_diagnostics(
            case=gpswf_case,
            modes=context.modes,
            retained=context.retained,
            target_nodes=setup.target_nodes,
            p_nodes=setup.measurement_nodes,
            mock_distances=setup.mock_distances,
            basis_matrix=context.target_basis,
            target_weights=setup.target_weights,
            component_data=component_data,
            coeffs=gpswf_coeffs,
            image=gpswf_rec,
            truth=context.truth,
            disk_mask=context.disk_mask,
        ))

        fourier_values, fourier_meta = reconstruct_fourier_cube_from_data(
            component_data,
            setup.target_nodes,
            setup.target_weights,
            context.grid_points,
            float(spec.k),
            C,
            half_side=half_side,
            bandwidth_factor=bandwidth_factor,
            max_modes=context.fourier_mode_cap,
            rcond=basis_lstsq_rcond,
        )
        fourier_rec = fourier_values.reshape(grid_size, grid_size)
        fourier_image = np.real(fourier_rec)
        fourier_title = f"Cube Fourier\nN={fourier_meta.get('fourier_modes', '?')}"
        reconstruction_panels.append((fourier_image, fourier_title, context.square_mask))
        _imshow(
            axes[row_index, 1],
            fourier_image,
            fourier_title,
            context.vmin,
            context.vmax,
        )
        fourier_case = _case_common(
            forward_row,
            experiment_number=experiment_number,
            row_index=row_index,
            column_index=1,
            method="cube_fourier",
            noise_level=noise_level,
            polarimetric_J=polarimetric_J,
        )
        fourier_case.update({
            "case_id": _case_id(experiment_number, spec, "fourier"),
            "support": "cube_half_side_1",
            "basis_modes": int(fourier_meta.get("fourier_modes", 0)),
            "basis_mode_fraction": float(fourier_mode_fraction),
            "basis_mode_cap": int(context.fourier_mode_cap),
            "basis_mode_hard_cap": int(context.N_cap_hard),
            "target_nodes": int(setup.target_nodes.shape[0]),
            "p_nodes": int(setup.target_nodes.shape[0]),
            **fourier_meta,
        })
        diagnostic_rows.append(_baseline_diagnostics(
            case=fourier_case,
            image=fourier_rec,
            truth=context.truth,
            valid_mask=context.square_mask,
        ))

        bessel_values, bessel_meta = reconstruct_ball_bessel_from_data(
            component_data,
            setup.target_nodes,
            setup.target_weights,
            context.grid_points,
            float(spec.k),
            C,
            quadrature_nodes=setup.target_nodes,
            quadrature_weights=setup.target_weights,
            bandwidth_factor=bandwidth_factor,
            max_modes=context.bessel_mode_cap,
            rcond=basis_lstsq_rcond,
        )
        bessel_rec = bessel_values.reshape(grid_size, grid_size)
        bessel_rec[~context.disk_mask] = 0.0
        bessel_image = np.real(bessel_rec)
        bessel_title = f"Ball Bessel\nN={bessel_meta.get('bessel_modes', '?')}"
        reconstruction_panels.append((bessel_image, bessel_title, context.disk_mask))
        _imshow(
            axes[row_index, 2],
            bessel_image,
            bessel_title,
            context.vmin,
            context.vmax,
        )
        bessel_case = _case_common(
            forward_row,
            experiment_number=experiment_number,
            row_index=row_index,
            column_index=2,
            method="ball_bessel",
            noise_level=noise_level,
            polarimetric_J=polarimetric_J,
        )
        bessel_case.update({
            "case_id": _case_id(experiment_number, spec, "bessel"),
            "support": "unit_ball",
            "basis_modes": int(bessel_meta.get("bessel_modes", 0)),
            "basis_mode_fraction": float(bessel_mode_fraction),
            "basis_mode_cap": int(context.bessel_mode_cap),
            "basis_mode_hard_cap": int(context.N_cap_hard),
            "target_nodes": int(setup.target_nodes.shape[0]),
            "p_nodes": int(setup.target_nodes.shape[0]),
            **bessel_meta,
        })
        diagnostic_rows.append(_baseline_diagnostics(
            case=bessel_case,
            image=bessel_rec,
            truth=context.truth,
            valid_mask=context.disk_mask,
        ))

        if include_dsm:
            dsm_values, dsm_meta = direct_sampling_component_indicator(
                component_data,
                setup.target_nodes,
                setup.target_weights,
                context.grid_points,
                C,
                normalize=True,
            )
            dsm_rec = dsm_values.reshape(grid_size, grid_size)
            dsm_rec[~context.disk_mask] = 0.0
            truth_scale = max(float(np.nanmax(np.abs(context.truth))), 1e-14)
            dsm_display = dsm_rec * truth_scale
            dsm_image = np.real(dsm_display)
            dsm_title = f"DSM\nnodes={dsm_meta.get('dsm_nodes', '?')}"
            reconstruction_panels.append((dsm_image, dsm_title, context.disk_mask))
            _imshow(
                axes[row_index, 3],
                dsm_image,
                dsm_title,
                context.vmin,
                context.vmax,
            )
            dsm_case = _case_common(
                forward_row,
                experiment_number=experiment_number,
                row_index=row_index,
                column_index=3,
                method="dsm",
                noise_level=noise_level,
                polarimetric_J=polarimetric_J,
            )
            dsm_case.update({
                "case_id": _case_id(experiment_number, spec, "dsm"),
                "support": "unit_ball",
                "basis_modes": int(dsm_meta.get("dsm_nodes", 0)),
                "target_nodes": int(setup.target_nodes.shape[0]),
                "p_nodes": int(setup.target_nodes.shape[0]),
                **dsm_meta,
            })
            diagnostic_rows.append(_baseline_diagnostics(
                case=dsm_case,
                image=dsm_display,
                truth=context.truth,
                valid_mask=context.disk_mask,
            ))

        axes[row_index, 0].set_ylabel(
            spec.label,
            fontsize=9,
            rotation=90,
            labelpad=18,
        )

    fig.suptitle(figure_title, fontsize=10)
    fig.savefig(config.out_dir / f"{output_stem}.png", dpi=200)
    plt.close(fig)

    adaptive_fig, adaptive_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.1 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        adaptive_axes = adaptive_axes[None, :]
    for flat_index, (image, title, valid_mask) in enumerate(reconstruction_panels):
        row_index, column_index = divmod(flat_index, n_cols)
        ax = adaptive_axes[row_index, column_index]
        valid_values = image[valid_mask]
        panel_vmin = float(np.nanmin(valid_values))
        panel_vmax = float(np.nanmax(valid_values))
        plotted = _imshow(ax, image, title, panel_vmin, panel_vmax)
        adaptive_fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.03)
        if column_index == 0:
            ax.set_ylabel(
                forward_rows[row_index].spec.label,
                fontsize=9,
                rotation=90,
                labelpad=18,
            )
    adaptive_fig.suptitle(figure_title, fontsize=10)
    adaptive_fig.savefig(
        config.out_dir / f"{output_stem}_individual_scale.png",
        dpi=200,
    )
    plt.close(adaptive_fig)

    prefix = f"exp{experiment_number}"
    write_diagnostics_csv(diagnostic_rows, config.out_dir / f"{prefix}_diagnostics.csv")
    save_diagnostics_npz(
        diagnostic_rows,
        config.out_dir / f"{prefix}_diagnostics_detail.npz",
    )
    plot_diagnostic_curves(
        diagnostic_rows,
        config.out_dir / f"{prefix}_diagnostic_curves.png",
        title=f"Experiment {experiment_number} diagnostics",
    )
    if gpswf_spatial_concentration_output is not None:
        concentration_path = config.out_dir / gpswf_spatial_concentration_output
        _plot_gpswf_spatial_concentration(
            forward_rows,
            context_cache,
            concentration_path,
        )
        print(f"Saved {concentration_path}")
    if gpswf_representative_modes_output is not None:
        available_k = sorted(context_cache)
        representative_k = (
            12.0
            if 12.0 in context_cache
            else available_k[len(available_k) // 2]
        )
        representative_row = next(
            row for row in forward_rows if float(row.spec.k) == representative_k
        )
        representative_path = config.out_dir / gpswf_representative_modes_output
        _plot_gpswf_representative_modes(
            context_cache[representative_k],
            float(representative_row.setup.params["C"]),
            representative_k,
            representative_path,
        )
        print(f"Saved {representative_path}")
    print(f"Saved {config.out_dir / f'{output_stem}.png'}")
    return make_table([{"experiment": experiment_number, "status": "ok"}])


def _case_id(
    experiment_number: int,
    spec: BasisComparisonRow,
    method: str,
) -> str:
    if experiment_number == 4:
        return f"k{spec.k:g}_{method}"
    return f"{spec.data_source}_{method}"


__all__ = [
    "BasisComparisonRow",
    "basis_n_per_axis",
    "basis_measure_dirs",
    "basis_row_params",
    "basis_row_params_quick",
    "run_basis_comparison",
]
