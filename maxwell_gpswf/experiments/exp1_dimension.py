#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1: Truncation dimension effects with a single cube block.

Layout: 2 rows x 3 cols.
  Row 1: N = 1, 21, 57
  Row 2: N = 71, 237, 496

Data: Full VIE far-field, finite-direction mock measurement mode, k=15,
far-field noise=0.2, and tensor contrast Q(x)=0.2 Q0 inside the cube.
Truncation: sort complete (ell, n) multiplets by |alpha| and use N as an
upper bound without splitting the m degeneracy.

The experiment also compares noiseless Full VIE and analytical Born data on
the same direction configurations.  This diagnostic is separate from the
noisy dimension-reconstruction panels.
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
    analytic_block_born_farfield_dataset,
    farfield_dataset_to_qhat,
    forward_solver_diagnostic_summary,
    full_vie_farfield_dataset,
    polarimetric_diagnostic_summary,
)


def _settings(quick: bool):
    """Return (n_measure, n_radial, n_angular, grid_size, n_per_axis, N_values, quad_order, r_eval)."""
    if quick:
        return 110, 10, 170, 51, 7, [1, 21, 57, 71, 237, 496], 100, 80
    return 974, 12, 230, 161, 23, [1, 21, 57, 71, 237, 496], 160, 120


def _relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    """Return ||values-reference|| / ||reference|| with a safe denominator."""
    denominator = max(float(np.linalg.norm(reference)), 1e-14)
    return float(np.linalg.norm(values - reference) / denominator)


def _complex_gain(values: np.ndarray, reference: np.ndarray) -> complex:
    """Least-squares complex gain mapping ``reference`` to ``values``."""
    denominator = np.vdot(reference, reference)
    if abs(denominator) <= 1e-14:
        return complex(np.nan, np.nan)
    return complex(np.vdot(reference, values) / denominator)


def _forward_model_diagnostics(
    *,
    full_dataset,
    born_dataset,
    full_qhat: np.ndarray,
    born_qhat: np.ndarray,
    target_nodes: np.ndarray,
    component_index: int,
    contrast_scale: float,
    k: float,
    n_per_axis: int,
    out_dir: Path,
) -> dict[str, float | int | str]:
    """Compare noiseless Full VIE and analytical Born data."""
    full_farfield = np.asarray(full_dataset.farfield_data, dtype=np.complex128)
    born_farfield = np.asarray(born_dataset.farfield_data, dtype=np.complex128)
    full_component = np.asarray(full_qhat[:, component_index], dtype=np.complex128)
    born_component = np.asarray(born_qhat[:, component_index], dtype=np.complex128)

    farfield_gain = _complex_gain(full_farfield, born_farfield)
    qhat_gain = _complex_gain(full_component, born_component)
    summary: dict[str, float | int | str] = {
        "experiment": 1,
        "case_id": "full_vie_vs_analytic_born",
        "k": float(k),
        "contrast_scale": float(contrast_scale),
        "n_per_axis": int(n_per_axis),
        "farfield_model_error_over_full": _relative_error(
            born_farfield, full_farfield
        ),
        "farfield_error_vs_born": _relative_error(full_farfield, born_farfield),
        "qhat_error_vs_born": _relative_error(full_qhat, born_qhat),
        "qhat11_error_vs_born": _relative_error(full_component, born_component),
        "farfield_gain_abs": float(abs(farfield_gain)),
        "farfield_gain_phase_deg": float(np.angle(farfield_gain, deg=True)),
        "qhat11_gain_abs": float(abs(qhat_gain)),
        "qhat11_gain_phase_deg": float(np.angle(qhat_gain, deg=True)),
        **forward_solver_diagnostic_summary(full_dataset),
    }

    radii = np.linalg.norm(np.asarray(target_nodes, dtype=float), axis=1)
    edges = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    born_amplitude = np.full(centers.shape, np.nan, dtype=float)
    full_amplitude = np.full(centers.shape, np.nan, dtype=float)
    relative_error = np.full(centers.shape, np.nan, dtype=float)
    cross_phase_deg = np.full(centers.shape, np.nan, dtype=float)
    bin_counts = np.zeros(centers.shape, dtype=int)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == centers.size - 1:
            mask = (radii >= lower) & (radii <= upper)
        else:
            mask = (radii >= lower) & (radii < upper)
        bin_counts[index] = int(np.count_nonzero(mask))
        if not np.any(mask):
            continue
        born_bin = born_component[mask]
        full_bin = full_component[mask]
        born_amplitude[index] = float(np.sqrt(np.mean(np.abs(born_bin) ** 2)))
        full_amplitude[index] = float(np.sqrt(np.mean(np.abs(full_bin) ** 2)))
        relative_error[index] = _relative_error(full_bin, born_bin)
        cross_phase_deg[index] = float(
            np.angle(_complex_gain(full_bin, born_bin), deg=True)
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(centers, born_amplitude, "o-", label="Analytic Born")
    axes[0, 0].plot(centers, full_amplitude, "s-", label="Full VIE")
    axes[0, 0].set_title(r"Radial RMS of $|\widehat{Q}_{11}|$")
    axes[0, 0].set_ylabel("RMS amplitude")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(centers, relative_error, "o-")
    axes[0, 1].set_title(r"Relative error of $\widehat{Q}_{11}$")
    axes[0, 1].set_ylabel("relative error")

    axes[1, 0].plot(centers, cross_phase_deg, "o-")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 0].set_title("Full VIE phase relative to Born")
    axes[1, 0].set_ylabel("phase (degrees)")

    axes[1, 1].axis("off")
    lines = [
        rf"$k={k:g}$, contrast scale $={contrast_scale:g}$",
        rf"Full-normalized far-field error $={summary['farfield_model_error_over_full']:.3f}$",
        rf"Born-normalized far-field error $={summary['farfield_error_vs_born']:.3f}$",
        rf"Full $\widehat{{Q}}$ error $={summary['qhat_error_vs_born']:.3f}$",
        rf"$\widehat{{Q}}_{{11}}$ error $={summary['qhat11_error_vs_born']:.3f}$",
        rf"$\widehat{{Q}}_{{11}}$ gain $={summary['qhat11_gain_abs']:.3f}$",
        rf"$\widehat{{Q}}_{{11}}$ phase $={summary['qhat11_gain_phase_deg']:.1f}^\circ$",
        rf"VIE grid $n_{{\mathrm{{axis}}}}={n_per_axis}$",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)
    for ax in axes.ravel()[:3]:
        ax.set_xlabel(r"Fourier radius $|p|$")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Experiment 1: Full VIE versus analytic Born", fontsize=12)
    fig.savefig(out_dir / "exp1_full_vs_born_diagnostics.png", dpi=200)
    plt.close(fig)

    write_diagnostics_csv(
        [summary], out_dir / "exp1_forward_model_diagnostics.csv"
    )
    np.savez_compressed(
        out_dir / "exp1_forward_model_diagnostics_detail.npz",
        p_radius=radii,
        qhat_full=full_qhat,
        qhat_born=born_qhat,
        qhat11_difference=full_component - born_component,
        radial_edges=edges,
        radial_centers=centers,
        radial_bin_counts=bin_counts,
        radial_born_rms=born_amplitude,
        radial_full_rms=full_amplitude,
        radial_relative_error=relative_error,
        radial_cross_phase_deg=cross_phase_deg,
    )
    return summary


def run_experiment(config: ExperimentConfig) -> Any:
    (requested_measure_dirs, n_radial, n_angular, grid_size,
     n_per_axis, N_values, quad_order, r_eval_count) = _settings(config.quick)

    k = 15.0; C = 2.0 * k
    kind = "full"; component_index = 0
    polarimetric_J = 6
    noise_level = 0.2
    cube_half_side = 0.4
    contrast_scale = 0.2

    # GPSWF params for k=15 (figure1/figure3 pattern)
    ell_max = 12
    n_modes_per_ell = 7
    K_val = 48

    rng = np.random.default_rng(config.seed + 100)

    # -- Target quadrature nodes --
    target_nodes, target_weights, _ = ball_quadrature_nodes(n_radial, n_angular)

    # -- Full VIE data with mock-matched direction pairs --
    vie_physical, vie_inc, vie_obs, vie_dist, data_info = generate_polarimetric_data_nodes(
        -target_nodes,
        requested_measure_dirs,
        polarimetric_J=polarimetric_J,
        tensor_kind=kind,
    )
    vie_nodes = -vie_physical

    ds = full_vie_farfield_dataset(
        "cube", vie_nodes, kind=kind, k=k, R=1.0,
        n_per_axis=n_per_axis, n_geometries=polarimetric_J,
        incident_dirs=vie_inc, obs_dirs=vie_obs,
        cube_half_side=cube_half_side,
        contrast_scale=contrast_scale)
    blocks = cube_phantom(
        center=(0.0, 0.0, 0.0),
        half_side=cube_half_side,
        amplitude=contrast_scale + 0.0j,
    )
    born_ds = analytic_block_born_farfield_dataset(
        blocks,
        vie_nodes,
        kind=kind,
        k=k,
        incident_dirs=vie_inc,
        obs_dirs=vie_obs,
    )
    full_qhat_clean = farfield_dataset_to_qhat(ds, kind=kind)
    born_qhat_clean = farfield_dataset_to_qhat(born_ds, kind=kind)
    forward_model_diagnostics = _forward_model_diagnostics(
        full_dataset=ds,
        born_dataset=born_ds,
        full_qhat=full_qhat_clean,
        born_qhat=born_qhat_clean,
        target_nodes=target_nodes,
        component_index=component_index,
        contrast_scale=contrast_scale,
        k=k,
        n_per_axis=n_per_axis,
        out_dir=config.out_dir,
    )
    rec_c = farfield_dataset_to_qhat(
        ds, kind=kind, noise_level=noise_level, rng=rng
    )
    comp_data = rec_c[:, component_index]
    polarimetric_diagnostics = polarimetric_diagnostic_summary(ds)
    forward_diagnostics = forward_solver_diagnostic_summary(ds)

    # -- Truth image (for vmin/vmax) --
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

    # -- Plot: six truncation dimensions in two rows and three columns --
    N_rows = 2; N_cols = 3
    fig, axes = plt.subplots(N_rows, N_cols,
                             figsize=(3.1 * N_cols, 3.1 * N_rows),
                             constrained_layout=True)
    axes = np.asarray(axes).reshape(N_rows, N_cols)
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
                "contrast_scale": float(contrast_scale),
                "n_radial": int(n_radial),
                "n_angular_requested": int(n_angular),
                "n_per_axis": int(n_per_axis),
                "n_geometries": int(polarimetric_J),
                "requested_measure_dirs": int(requested_measure_dirs),
                "candidate_count": int(data_info["candidate_count"]),
                "data_mode": "mock",
                "data_source": "full_vie",
                "shape": "cube",
                **polarimetric_diagnostics,
                **forward_diagnostics,
                "farfield_model_error_over_full": float(
                    forward_model_diagnostics["farfield_model_error_over_full"]
                ),
                "farfield_error_vs_born": float(
                    forward_model_diagnostics["farfield_error_vs_born"]
                ),
                "qhat_error_vs_born": float(
                    forward_model_diagnostics["qhat_error_vs_born"]
                ),
                "qhat11_error_vs_born": float(
                    forward_model_diagnostics["qhat11_error_vs_born"]
                ),
                "qhat11_gain_abs": float(
                    forward_model_diagnostics["qhat11_gain_abs"]
                ),
                "qhat11_gain_phase_deg": float(
                    forward_model_diagnostics["qhat11_gain_phase_deg"]
                ),
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
    adaptive_axes = np.asarray(adaptive_axes).reshape(N_rows, N_cols)
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
    print(f"Saved {config.out_dir / 'exp1_full_vs_born_diagnostics.png'}")
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
