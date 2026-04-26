#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

from three_small_obstacles_joint_gn_random_centers import (
    PI2,
    BoundaryGeometry,
    add_relative_noise,
    build_single_layer_matrix,
    dense_boundary_points,
    obstacle_param_slice,
    parse_float_list,
    params_to_geometry,
    plane_wave,
    single_layer_farfield_operator,
)

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TargetCase:
    name: str
    label: str
    params: Array
    n_obstacles: int
    n_boundary: int
    grid_extent: float


def make_star_block(
    center: tuple[float, float],
    radius: float,
    a2c: float = 0.0,
    a2s: float = 0.0,
    a3c: float = 0.0,
    a3s: float = 0.0,
) -> Array:
    return np.array([center[0], center[1], radius, a2c, a2s, a3c, a3s], dtype=float)


def make_cases() -> list[TargetCase]:
    one_small = np.concatenate([
        make_star_block((0.02, -0.01), 0.045, 0.10, -0.06, 0.04, 0.02),
    ])
    three_small = np.concatenate([
        make_star_block((-0.16, -0.08), 0.043, 0.10, -0.05, 0.04, 0.02),
        make_star_block((0.12, -0.02), 0.046, -0.08, 0.08, -0.04, 0.03),
        make_star_block((0.00, 0.15), 0.044, 0.06, 0.09, 0.04, -0.05),
    ])
    one_large = np.concatenate([
        make_star_block((0.02, 0.00), 0.18, 0.22, -0.08, 0.07, 0.05),
    ])
    return [
        TargetCase("one_small", "1 small target", one_small, 1, 80, 0.45),
        TargetCase("three_small", "3 small targets", three_small, 3, 48, 0.50),
        TargetCase("one_large", "1 large target", one_large, 1, 180, 0.60),
    ]


def limited_aperture_angles(center: float, alpha: float, n_obs: int) -> Array:
    if not (0.0 < alpha <= math.pi):
        raise ValueError("--alpha must be in (0, pi]")
    if n_obs < 2:
        raise ValueError("--n-obs must be at least 2 for a finite aperture")
    if abs(alpha - math.pi) < 1e-14:
        return np.linspace(0.0, PI2, n_obs, endpoint=False)
    return center + np.linspace(-alpha, alpha, n_obs)


def solve_forward_farfield_variable(
    params: Array,
    n_obstacles: int,
    k: float,
    n_per_obstacle: int,
    incident_angles: Array,
    obs_angles: Array,
) -> CArray:
    geom: BoundaryGeometry = params_to_geometry(params, n_per_obstacle, n_obstacles=n_obstacles)
    A = build_single_layer_matrix(geom, k)
    Ainf = single_layer_farfield_operator(geom, k, obs_angles)
    eye = np.eye(A.shape[0], dtype=complex)
    stab = 1e-12 * max(np.linalg.norm(A, ord=2), 1.0)
    farfield = np.empty((len(obs_angles), len(incident_angles)), dtype=complex)
    for j, ang in enumerate(incident_angles):
        d = np.array([math.cos(float(ang)), math.sin(float(ang))], dtype=float)
        rhs = -plane_wave(geom.x, k, d)
        density = solve(A + stab * eye, rhs, assume_a="gen")
        farfield[:, j] = Ainf @ density
    return farfield


def direct_sampling_indicator_limited_aperture(
    farfield_matrix: CArray,
    k: float,
    obs_angles: Array,
    incident_angles: Array,
    x_grid: Array,
    y_grid: Array,
    aperture_length: float,
    power: float = 1.0,
) -> Array:
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    obs_weight = aperture_length / max(len(obs_angles) - 1, 1)
    inc_weight = PI2 / len(incident_angles)
    phase = np.exp(1j * k * (xhat @ pts.T))
    backpropagated = obs_weight * (farfield_matrix.T @ phase)
    indicator = inc_weight * np.sum(np.abs(backpropagated) ** float(power), axis=0)
    indicator = indicator.reshape(X.shape).astype(float, copy=False)
    indicator /= max(float(np.max(indicator)), 1e-14)
    return indicator


def boundary_sets(case: TargetCase) -> list[Array]:
    return [
        dense_boundary_points(case.params[obstacle_param_slice(j)], n=500)
        for j in range(case.n_obstacles)
    ]


def save_case_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, case: TargetCase, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    for pts in boundary_sets(case):
        ax.plot(pts[:, 0], pts[:, 1], "k--", lw=1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("normalized indicator")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary_plot(
    path: Path,
    images: Sequence[Array],
    grids: Sequence[tuple[Array, Array]],
    cases: Sequence[TargetCase],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(15.0, 4.7), constrained_layout=True)
    for ax, image, (x_grid, y_grid), case in zip(axes, images, grids, cases):
        m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
        for pts in boundary_sets(case):
            ax.plot(pts[:, 0], pts[:, 1], "k--", lw=1.1)
        ax.set_aspect("equal")
        ax.set_title(case.label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(m, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("normalized indicator")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fixed-frequency limited-aperture direct sampling images for small and large sound-soft targets."
    )
    p.add_argument("--out-dir", type=str, default="outputs_limited_aperture_direct_sampling_targets")
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=math.pi / 3.0, help="observation aperture half-angle in radians")
    p.add_argument("--aperture-center", type=float, default=0.0, help="central observation direction theta0 in radians")
    p.add_argument("--n-obs", type=int, default=81)
    p.add_argument(
        "--incident-angles",
        type=str,
        default="0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438",
    )
    p.add_argument("--grid-size", type=int, default=241)
    p.add_argument("--noise-level", type=float, default=0.03)
    p.add_argument("--indicator-power", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = float(args.k)
    alpha = float(args.alpha)
    aperture_length = PI2 if abs(alpha - math.pi) < 1e-14 else 2.0 * alpha
    obs_angles = limited_aperture_angles(float(args.aperture_center), alpha, int(args.n_obs))
    incident_angles = parse_float_list(args.incident_angles)
    rng = np.random.default_rng(int(args.seed))

    cases = make_cases()
    clean_images: list[Array] = []
    noisy_images: list[Array] = []
    grids: list[tuple[Array, Array]] = []
    metadata_cases = []

    for case in cases:
        x_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        y_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        grids.append((x_grid, y_grid))

        farfield_clean = solve_forward_farfield_variable(
            case.params,
            case.n_obstacles,
            k,
            case.n_boundary,
            incident_angles,
            obs_angles,
        )
        farfield_noisy = add_relative_noise(farfield_clean, float(args.noise_level), rng)

        image_clean = direct_sampling_indicator_limited_aperture(
            farfield_clean,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            aperture_length,
            power=float(args.indicator_power),
        )
        image_noisy = direct_sampling_indicator_limited_aperture(
            farfield_noisy,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            aperture_length,
            power=float(args.indicator_power),
        )
        clean_images.append(image_clean)
        noisy_images.append(image_noisy)

        save_case_plot(
            out_dir / f"{case.name}_clean.png",
            image_clean,
            x_grid,
            y_grid,
            case,
            f"{case.label}, clean data",
        )
        save_case_plot(
            out_dir / f"{case.name}_noisy.png",
            image_noisy,
            x_grid,
            y_grid,
            case,
            f"{case.label}, noise={float(args.noise_level):.2f}",
        )
        np.savez_compressed(
            out_dir / f"{case.name}_result.npz",
            params=case.params,
            farfield_clean=farfield_clean,
            farfield_noisy=farfield_noisy,
            image_clean=image_clean,
            image_noisy=image_noisy,
            x_grid=x_grid,
            y_grid=y_grid,
            obs_angles=obs_angles,
            incident_angles=incident_angles,
            k=k,
            alpha=alpha,
            aperture_center=float(args.aperture_center),
        )
        metadata_cases.append(
            {
                "name": case.name,
                "label": case.label,
                "n_obstacles": case.n_obstacles,
                "n_boundary_per_obstacle": case.n_boundary,
                "grid_extent": case.grid_extent,
                "params": case.params.tolist(),
                "clean_plot": str(out_dir / f"{case.name}_clean.png"),
                "noisy_plot": str(out_dir / f"{case.name}_noisy.png"),
                "result_npz": str(out_dir / f"{case.name}_result.npz"),
            }
        )

    summary_title = (
        f"Limited-aperture direct sampling, k={k:g}, "
        f"theta0={float(args.aperture_center):.2f}, alpha={alpha:.2f}"
    )
    save_summary_plot(out_dir / "summary_clean.png", clean_images, grids, cases, summary_title + ", clean")
    save_summary_plot(out_dir / "summary_noisy.png", noisy_images, grids, cases, summary_title + f", noise={float(args.noise_level):.2f}")

    metadata = {
        "method": "limited-aperture multi-direction orthogonality/direct sampling",
        "reference_indicator": "mu(y,k,d)=|int_Gamma_obs exp(i*k*xhat dot y) u_inf(xhat,d,k) ds(xhat)|; summed over incident directions",
        "k": k,
        "alpha": alpha,
        "aperture_center": float(args.aperture_center),
        "aperture_length": aperture_length,
        "n_obs": int(args.n_obs),
        "incident_angles": incident_angles.tolist(),
        "noise_level": float(args.noise_level),
        "indicator_power": float(args.indicator_power),
        "cases": metadata_cases,
        "summary_clean": str(out_dir / "summary_clean.png"),
        "summary_noisy": str(out_dir / "summary_noisy.png"),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
