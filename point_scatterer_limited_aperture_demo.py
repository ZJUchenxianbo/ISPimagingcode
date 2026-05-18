#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-frequency limited-aperture imaging for point scatterers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampling_imaging import (
    PI2,
    Array,
    add_relative_complex_noise,
    aperture_angles,
    aperture_measure,
    direct_sampling_indicator,
    plot_indicator_image,
    point_scatterer_farfield,
    safe_slug,
)


def plot_summary(
    out_path: Path,
    clean_images: list[Array],
    noisy_images: list[Array],
    aperture_labels: list[str],
    x_grid: Array,
    y_grid: Array,
    points: Array,
    k: float,
    aperture_center: float,
    noise_level: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    rows = [(clean_images, "clean"), (noisy_images, f"{noise_level:.0%} noise")]
    for row_idx, (images, row_label) in enumerate(rows):
        for col_idx, (image, aperture_label) in enumerate(zip(images, aperture_labels)):
            ax = axes[row_idx, col_idx]
            im = plot_indicator_image(
                ax,
                image,
                x_grid,
                y_grid,
                title=f"{aperture_label}, {row_label}",
            )
            ax.scatter(points[:, 0], points[:, 1], c="black", marker="x", s=62, linewidths=1.8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="normalized indicator")
    fig.suptitle(f"Point scatterers, fixed frequency k={k:g}, observation center theta0={aperture_center:g}")
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-frequency limited-aperture imaging for point scatterers.")
    p.add_argument("--out-dir", type=str, default="outputs_point_scatterers_limited_aperture")
    p.add_argument("--k", type=float, default=12.0)
    p.add_argument("--n-obs", type=int, default=121)
    p.add_argument("--noise-level", type=float, default=0.10)
    p.add_argument("--aperture-center", type=float, default=0.0)
    p.add_argument("--grid-extent", type=float, default=0.42)
    p.add_argument("--grid-size", type=int, default=321)
    p.add_argument("--block-size", type=int, default=32768)
    p.add_argument("--seed", type=int, default=20260518)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = float(args.k)
    n_obs = int(args.n_obs)
    noise_level = float(args.noise_level)
    aperture_center = float(args.aperture_center)
    incident_angles = np.linspace(0.0, PI2, 8, endpoint=False)
    points = np.array(
        [
            [-0.20, -0.10],
            [0.13, -0.03],
            [0.02, 0.18],
        ],
        dtype=float,
    )
    strengths = np.array([1.0 + 0.0j, 0.85 * np.exp(0.4j), 1.15 * np.exp(-0.7j)], dtype=complex)

    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))

    apertures = [
        ("full aperture", math.pi),
        ("limited alpha=pi/2", math.pi / 2.0),
        ("limited alpha=pi/3", math.pi / 3.0),
    ]

    clean_images: list[Array] = []
    noisy_images: list[Array] = []
    metadata_apertures = []
    for idx, (label, alpha) in enumerate(apertures):
        aperture_length = aperture_measure(alpha)
        obs_angles = aperture_angles(aperture_center, alpha, n_obs)
        farfield_clean = point_scatterer_farfield(points, strengths, k, incident_angles, obs_angles)
        farfield_noisy = add_relative_complex_noise(farfield_clean, noise_level, int(args.seed) + idx)

        image_clean = direct_sampling_indicator(
            farfield_clean,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            aperture_length,
            block_size=int(args.block_size),
        )
        image_noisy = direct_sampling_indicator(
            farfield_noisy,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            aperture_length,
            block_size=int(args.block_size),
        )
        clean_images.append(image_clean)
        noisy_images.append(image_noisy)

        safe_label = safe_slug([label])
        np.savez_compressed(
            out_dir / f"{safe_label}.npz",
            points=points,
            strengths=strengths,
            k=k,
            alpha=alpha,
            aperture_center=aperture_center,
            obs_angles=obs_angles,
            incident_angles=incident_angles,
            farfield_clean=farfield_clean,
            farfield_noisy=farfield_noisy,
            image_clean=image_clean,
            image_noisy=image_noisy,
            x_grid=x_grid,
            y_grid=y_grid,
        )
        metadata_apertures.append(
            {
                "label": label,
                "alpha": alpha,
                "aperture_length": aperture_length,
                "npz": str(out_dir / f"{safe_label}.npz"),
            }
        )

    summary_path = out_dir / "point_scatterers_summary.png"
    plot_summary(
        summary_path,
        clean_images,
        noisy_images,
        [label for label, _ in apertures],
        x_grid,
        y_grid,
        points,
        k,
        aperture_center,
        noise_level,
    )

    metadata = {
        "model": "u_inf(xhat,d)=sum_j q_j exp(-i*k*xhat dot z_j) exp(i*k*d dot z_j)",
        "indicator": "sum_d |int_Gamma u_inf(xhat,d) exp(i*k*xhat dot y) ds(xhat)|",
        "summary_plot": str(summary_path),
        "k": k,
        "noise_level": noise_level,
        "n_obs": n_obs,
        "aperture_center": aperture_center,
        "block_size": int(args.block_size),
        "incident_angles": incident_angles.tolist(),
        "points": points.tolist(),
        "strengths_real_imag": [[float(np.real(q)), float(np.imag(q))] for q in strengths],
        "apertures": metadata_apertures,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
