#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""有限孔径条件下的直接采样/正交采样成像实验。

脚本可比较多类目标，例如小目标、小目标簇和较大目标。
具体运行哪些目标由 --cases 参数控制。

与全孔径观测不同，这里观测方向只覆盖一个角度区间
    [aperture_center - alpha, aperture_center + alpha]
因此成像会出现方向性模糊或分辨率下降。脚本会分别保存无噪声和有噪声指标图。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from common.targets import (
    ObstacleTargetCase as TargetCase,
    limited_aperture_obstacle_cases,
    parse_case_names,
    plot_obstacle_boundaries,
)
from common.direct_sampling import compute_direct_sampling_result
from common.scattering import parse_float_list
from common.sampling import (
    aperture_angles,
    aperture_measure,
)

Array = NDArray[np.float64]


def save_case_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, case: TargetCase, title: str) -> None:
    """保存单个案例的指标图，并叠加真实目标边界。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    plot_obstacle_boundaries(ax, case.params, case.n_obstacles, "k--", lw=1.25, n=500)
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
    """把多个目标案例并排画在同一张总览图中。"""
    fig, axes = plt.subplots(1, len(cases), figsize=(15.0, 4.7), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    m = None
    for ax, image, (x_grid, y_grid), case in zip(axes_arr, images, grids, cases):
        m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
        plot_obstacle_boundaries(ax, case.params, case.n_obstacles, "k--", lw=1.1, n=500)
        ax.set_aspect("equal")
        ax.set_title(case.label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(m, ax=axes_arr.ravel().tolist(), shrink=0.85)
    cbar.set_label("normalized indicator")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """主程序：对每个目标案例生成有限孔径远场数据并计算直接采样指标。"""
    p = argparse.ArgumentParser(
        description="Fixed-frequency limited-aperture direct sampling images for small and large sound-soft targets."
    )

    # ---------- 命令行参数 ----------
    p.add_argument("--out-dir", type=str, default="outputs_obstacle_limited_aperture_imaging")
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
    p.add_argument("--noise-levels", type=str, default="0.05,0.10,0.20")
    p.add_argument("--noise-level", type=float, default=None, help="deprecated single-noise override")
    p.add_argument("--indicator-power", type=float, default=1.0)
    p.add_argument("--block-size", type=int, default=32768)
    p.add_argument(
        "--cases",
        type=str,
        default="all",
        help="comma-separated target cases: all, small_target, small_cluster, large_target",
    )
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()

    # 创建输出目录。
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 观测孔径、入射方向和随机数 ----------
    k = float(args.k)
    alpha = float(args.alpha)

    # alpha=pi 表示全孔径，孔径长度为 2pi；否则有限孔径长度为 2alpha。
    aperture_length = aperture_measure(alpha)
    obs_angles = aperture_angles(float(args.aperture_center), alpha, int(args.n_obs))
    incident_angles = parse_float_list(args.incident_angles)
    noise_levels = np.asarray([float(args.noise_level)], dtype=float) if args.noise_level is not None else parse_float_list(args.noise_levels)

    cases = limited_aperture_obstacle_cases(parse_case_names(args.cases))
    clean_images: list[Array] = []
    noisy_images_by_noise: dict[float, list[Array]] = {float(noise): [] for noise in noise_levels}
    grids: list[tuple[Array, Array]] = []
    metadata_cases = []

    for case in cases:
        # 每个案例可以有不同成像范围，因此单独生成网格。
        x_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        y_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        grids.append((x_grid, y_grid))

        # 先生成无噪声远场，再添加指定相对噪声，并计算对应直接采样指标。
        result = compute_direct_sampling_result(
            case.params,
            case.n_obstacles,
            case.n_boundary,
            k,
            incident_angles,
            obs_angles,
            x_grid,
            y_grid,
            aperture_length,
            noise_levels,
            int(args.seed),
            indicator_power=float(args.indicator_power),
            block_size=int(args.block_size),
        )
        image_clean = result.image_clean

        # 保存单案例图片。
        save_case_plot(
            out_dir / f"{case.name}_clean.png",
            image_clean,
            x_grid,
            y_grid,
            case,
            f"{case.label}, clean data",
        )
        noisy_plot_paths = []
        for noise_level, image_noisy in zip(noise_levels, result.image_noisy_list):
            noisy_plot = out_dir / f"{case.name}_noisy_{float(noise_level):.2f}.png"
            save_case_plot(
                noisy_plot,
                image_noisy,
                x_grid,
                y_grid,
                case,
                f"{case.label}, noise={float(noise_level):.2f}",
            )
            noisy_images_by_noise[float(noise_level)].append(image_noisy)
            noisy_plot_paths.append(str(noisy_plot))

        # 保存该案例所有核心数组。
        np.savez_compressed(
            out_dir / f"{case.name}_result.npz",
            params=case.params,
            farfield_clean=result.farfield_clean,
            farfield_noisy=np.stack(result.farfield_noisy_list, axis=0),
            image_clean=image_clean,
            image_noisy=np.stack(result.image_noisy_list, axis=0),
            x_grid=x_grid,
            y_grid=y_grid,
            obs_angles=obs_angles,
            incident_angles=incident_angles,
            k=k,
            alpha=alpha,
            noise_levels=noise_levels,
            aperture_center=float(args.aperture_center),
            n_obstacles=case.n_obstacles,
        )
        clean_images.append(image_clean)

        # 记录该案例的元数据和输出路径。
        metadata_cases.append(
            {
                "name": case.name,
                "label": case.label,
                "n_obstacles": case.n_obstacles,
                "n_boundary_per_obstacle": case.n_boundary,
                "grid_extent": case.grid_extent,
                "params": case.params.tolist(),
                "clean_plot": str(out_dir / f"{case.name}_clean.png"),
                "noisy_plots": noisy_plot_paths,
                "result_npz": str(out_dir / f"{case.name}_result.npz"),
            }
        )

    summary_title = (
        f"Limited-aperture direct sampling, k={k:g}, "
        f"theta0={float(args.aperture_center):.2f}, alpha={alpha:.2f}"
    )

    # 保存无噪声/有噪声总览图。
    save_summary_plot(out_dir / "summary_clean.png", clean_images, grids, cases, summary_title + ", clean")
    summary_noisy_paths = []
    for noise_level in noise_levels:
        summary_noisy = out_dir / f"summary_noisy_{float(noise_level):.2f}.png"
        save_summary_plot(
            summary_noisy,
            noisy_images_by_noise[float(noise_level)],
            grids,
            cases,
            summary_title + f", noise={float(noise_level):.2f}",
        )
        summary_noisy_paths.append(str(summary_noisy))

    # 保存整次实验的元数据。
    metadata = {
        "method": "limited-aperture multi-direction orthogonality/direct sampling",
        "reference_indicator": "mu(y,k,d)=|int_Gamma_obs exp(i*k*xhat dot y) u_inf(xhat,d,k) ds(xhat)|; summed over incident directions",
        "k": k,
        "alpha": alpha,
        "aperture_center": float(args.aperture_center),
        "aperture_length": aperture_length,
        "n_obs": int(args.n_obs),
        "incident_angles": incident_angles.tolist(),
        "noise_levels": noise_levels.tolist(),
        "indicator_power": float(args.indicator_power),
        "block_size": int(args.block_size),
        "selected_cases": [case.name for case in cases],
        "cases": metadata_cases,
        "summary_clean": str(out_dir / "summary_clean.png"),
        "summary_noisy": summary_noisy_paths,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 终端打印摘要，方便运行后直接查看输出文件位置。
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    # 作为脚本运行时执行实验；被 import 时只提供函数。
    main()
