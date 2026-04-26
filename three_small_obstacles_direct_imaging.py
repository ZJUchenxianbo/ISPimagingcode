#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from three_small_obstacles_joint_gn_random_centers import (
    PI2,
    add_relative_noise,
    dense_boundary_points,
    generate_random_centers,
    obstacle_param_slice,
    parse_float_list,
    solve_forward_farfield,
)

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def build_true_params(args: argparse.Namespace) -> tuple[Array, Array]:
    center_extent = float(args.center_extent)
    rng_cent = np.random.default_rng(int(args.seed))
    centers_true = generate_random_centers(float(args.spacing), rng_cent, center_extent, float(args.min_gap))

    coeffs_true = [
        np.array([float(args.radius), float(args.true1_a2c), float(args.true1_a2s), float(args.true1_a3c), float(args.true1_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true2_a2c), float(args.true2_a2s), float(args.true2_a3c), float(args.true2_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true3_a2c), float(args.true3_a2s), float(args.true3_a3c), float(args.true3_a3s)], dtype=float),
    ]
    p_true = np.concatenate([np.concatenate([centers_true[j], coeffs_true[j]]) for j in range(3)]).astype(float)
    return p_true, centers_true


def orthogonality_sampling_indicator_md(
    farfield_matrix: CArray,
    k: float,
    obs_angles: Array,
    incident_angles: Array,
    x_grid: Array,
    y_grid: Array,
    power: float = 1.0,
) -> Array:
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    obs_weight = PI2 / len(obs_angles)
    inc_weight = PI2 / len(incident_angles)
    phase = np.exp(1j * k * (xhat @ pts.T))
    reduced_fields = obs_weight * (farfield_matrix.T @ phase)
    indicator = inc_weight * np.sum(np.abs(reduced_fields) ** float(power), axis=0)
    indicator = indicator.reshape(X.shape).astype(float, copy=False)
    indicator /= max(np.max(indicator), 1e-14)
    return indicator


def save_imaging_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, p_true: Array, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r")
    for j in range(3):
        pts = dense_boundary_points(p_true[obstacle_param_slice(j)])
        ax.plot(pts[:, 0], pts[:, 1], "k--", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("normalized indicator")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Direct imaging for three small obstacles using multi-direction orthogonality sampling"
    )
    p.add_argument("--out-dir", type=str, default="outputs_three_small_obstacles_direct_imaging")
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--radius", type=float, default=0.045)
    p.add_argument("--spacing", type=float, default=0.18)
    p.add_argument("--noise-level", type=float, default=0.05)
    p.add_argument(
        "--incident-angles",
        type=str,
        default="0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438",
    )
    p.add_argument("--n-per-obstacle", type=int, default=10)
    p.add_argument("--n-obs", type=int, default=72)
    p.add_argument("--grid-extent", type=float, default=0.45)
    p.add_argument("--grid-size", type=int, default=201)
    p.add_argument("--center-extent", type=float, default=0.22)
    p.add_argument("--min-gap", type=float, default=0.008)
    p.add_argument("--indicator-power", type=float, default=1.0)
    p.add_argument("--true1-a2c", type=float, default=0.12)
    p.add_argument("--true1-a2s", type=float, default=-0.08)
    p.add_argument("--true1-a3c", type=float, default=0.06)
    p.add_argument("--true1-a3s", type=float, default=0.03)
    p.add_argument("--true2-a2c", type=float, default=-0.10)
    p.add_argument("--true2-a2s", type=float, default=0.09)
    p.add_argument("--true2-a3c", type=float, default=-0.05)
    p.add_argument("--true2-a3s", type=float, default=0.04)
    p.add_argument("--true3-a2c", type=float, default=0.08)
    p.add_argument("--true3-a2s", type=float, default=0.10)
    p.add_argument("--true3-a3c", type=float, default=0.05)
    p.add_argument("--true3-a3s", type=float, default=-0.06)
    p.add_argument("--seed", type=int, default=24680)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = float(args.k)
    noise_level = float(args.noise_level)
    incident_angles = parse_float_list(args.incident_angles)
    obs_angles = np.linspace(0.0, PI2, int(args.n_obs), endpoint=False)
    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))

    p_true, centers_true = build_true_params(args)
    farfield_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)
    rng_noise = np.random.default_rng(int(args.seed) + 999)
    farfield_noisy = add_relative_noise(farfield_clean, noise_level, rng_noise)

    image_clean = orthogonality_sampling_indicator_md(
        farfield_clean,
        k,
        obs_angles,
        incident_angles,
        x_grid,
        y_grid,
        power=float(args.indicator_power),
    )
    image_noisy = orthogonality_sampling_indicator_md(
        farfield_noisy,
        k,
        obs_angles,
        incident_angles,
        x_grid,
        y_grid,
        power=float(args.indicator_power),
    )

    save_imaging_plot(
        out_dir / "direct_imaging_clean.png",
        image_clean,
        x_grid,
        y_grid,
        p_true,
        title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), clean data",
    )
    save_imaging_plot(
        out_dir / "direct_imaging_noisy.png",
        image_noisy,
        x_grid,
        y_grid,
        p_true,
        title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), noise={noise_level:.2f}",
    )

    np.savez_compressed(
        out_dir / "direct_imaging_result.npz",
        p_true=p_true,
        centers_true=centers_true,
        farfield_clean=farfield_clean,
        farfield_noisy=farfield_noisy,
        image_clean=image_clean,
        image_noisy=image_noisy,
        x_grid=x_grid,
        y_grid=y_grid,
        obs_angles=obs_angles,
        incident_angles=incident_angles,
        k=k,
        noise_level=noise_level,
    )

    metadata = {
        "method": "multi-direction orthogonality sampling",
        "indicator_formula": "mu_MD(y,k)=sum_d |sum_xhat exp(i*k*(xhat dot y)) u_inf(xhat,d,k)|",
        "indicator_power": float(args.indicator_power),
        "colormap": "RdYlBu_r",
        "k": k,
        "noise_level": noise_level,
        "seed": int(args.seed),
        "centers_true": centers_true.tolist(),
        "output_clean_plot": str(out_dir / "direct_imaging_clean.png"),
        "output_noisy_plot": str(out_dir / "direct_imaging_noisy.png"),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
