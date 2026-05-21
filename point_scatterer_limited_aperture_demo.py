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
    direct_sampling_indicators,
    plot_indicator_image,
    point_scatterer_farfield,
    safe_slug,
)
from unet_imaging import (
    PointScattererUNetConfig,
    predict_unet_images,
    save_unet_checkpoint,
    train_point_scatterer_unet,
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
    method_label: str = "Direct sampling",
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
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="normalized indicator") # type: ignore
    fig.suptitle(
        f"{method_label}: point scatterers, fixed frequency k={k:g}, observation center theta0={aperture_center:g}"
    )
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-frequency limited-aperture imaging for point scatterers.")
    p.add_argument("--out-dir", type=str, default="outputs_point_scatterers_limited_aperture")
    p.add_argument("--k", type=float, default=2*PI2)
    p.add_argument("--n-obs", type=int, default=121)
    p.add_argument("--noise-level", type=float, default=0.10)
    p.add_argument("--aperture-center", type=float, default=0.0)
    p.add_argument("--grid-extent", type=float, default=0.42)
    p.add_argument("--grid-size", type=int, default=321)
    p.add_argument("--block-size", type=int, default=32768)
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--no-unet", action="store_true", help="Skip the learned U-Net postprocess.")
    p.add_argument("--unet-size", type=int, default=96)
    p.add_argument("--unet-samples", type=int, default=128)
    p.add_argument("--unet-epochs", type=int, default=12)
    p.add_argument("--unet-batch-size", type=int, default=8)
    p.add_argument("--unet-base-channels", type=int, default=8)
    p.add_argument("--unet-lr", type=float, default=1e-3)
    p.add_argument("--unet-train-n-obs", type=int, default=61)
    p.add_argument("--unet-target-sigma", type=float, default=0.035)
    p.add_argument("--unet-seed", type=int, default=20260519)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = float(args.k)
    n_obs = int(args.n_obs)
    noise_level = float(args.noise_level)
    aperture_center = float(args.aperture_center)
    incident_angles = np.linspace(math.pi/2,3*math.pi/2,3,endpoint=False)
    points = np.array(
        [
            [-0.20, -0.10],
            [0.30, -0.10],
            [-0.20,0.20],
        ],
        dtype=float,
    )
    strengths = np.array([1.0 + 0.0j, 0.85 * np.exp(0.4j),1.15*np.exp(-0.7j)], dtype=complex)

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
        print(f"computing {label} ({idx + 1}/{len(apertures)})...", flush=True)
        aperture_length = aperture_measure(alpha)
        obs_angles = aperture_angles(aperture_center, alpha, n_obs)
        farfield_clean = point_scatterer_farfield(points, strengths, k, incident_angles, obs_angles)
        farfield_noisy = add_relative_complex_noise(farfield_clean, noise_level, int(args.seed) + idx)

        image_clean, image_noisy = direct_sampling_indicators(
            [farfield_clean, farfield_noisy],
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
        "Direct sampling",
    )

    unet_summary_path: Path | None = None
    unet_history: list[float] | None = None
    if not args.no_unet:
        print("training U-Net from synthetic direct-sampling images...", flush=True)
        unet_config = PointScattererUNetConfig(
            k=k,
            incident_angles=incident_angles,
            aperture_center=aperture_center,
            aperture_half_widths=tuple(alpha for _, alpha in apertures),
            noise_level=noise_level,
            grid_extent=float(args.grid_extent),
            grid_size=int(args.unet_size),
            n_obs=int(args.unet_train_n_obs),
            n_samples=int(args.unet_samples),
            max_points=max(1, int(points.shape[0])),
            target_sigma=float(args.unet_target_sigma),
            block_size=min(int(args.block_size), 8192),
            seed=int(args.unet_seed),
        )
        unet_model, unet_history = train_point_scatterer_unet(
            unet_config,
            epochs=int(args.unet_epochs),
            batch_size=int(args.unet_batch_size),
            learning_rate=float(args.unet_lr),
            base_channels=int(args.unet_base_channels),
        )
        learned_clean_images = predict_unet_images(
            unet_model,
            clean_images,
            network_size=int(args.unet_size),
            batch_size=int(args.unet_batch_size),
        )
        learned_noisy_images = predict_unet_images(
            unet_model,
            noisy_images,
            network_size=int(args.unet_size),
            batch_size=int(args.unet_batch_size),
        )
        unet_summary_path = out_dir / "point_scatterers_unet_summary.png"
        plot_summary(
            unet_summary_path,
            learned_clean_images,
            learned_noisy_images,
            [label for label, _ in apertures],
            x_grid,
            y_grid,
            points,
            k,
            aperture_center,
            noise_level,
            "U-Net learned postprocess",
        )
        np.savez_compressed(
            out_dir / "unet_images.npz",
            learned_clean_images=np.stack(learned_clean_images, axis=0),
            learned_noisy_images=np.stack(learned_noisy_images, axis=0),
            history=np.asarray(unet_history, dtype=float),
            x_grid=x_grid,
            y_grid=y_grid,
        )
        save_unet_checkpoint(
            out_dir / "point_scatterer_unet_model.pt",
            unet_model,
            unet_history,
            {
                "input": "direct-sampling indicator",
                "target": "Gaussian point-support heatmap",
                "grid_size": int(args.unet_size),
                "n_samples": int(args.unet_samples),
                "epochs": int(args.unet_epochs),
                "base_channels": int(args.unet_base_channels),
                "target_sigma": float(args.unet_target_sigma),
            },
        )

    metadata = {
        "model": "u_inf(xhat,d)=sum_j q_j exp(-i*k*xhat dot z_j) exp(i*k*d dot z_j)",
        "indicator": "sum_d |int_Gamma u_inf(xhat,d) exp(i*k*xhat dot y) ds(xhat)|",
        "summary_plot": str(summary_path),
        "unet_summary_plot": str(unet_summary_path) if unet_summary_path is not None else None,
        "unet_history": unet_history,
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
