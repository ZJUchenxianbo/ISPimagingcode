#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三小障碍物的多方向正交采样直接成像实验。

这个脚本只做“定性成像”：先构造三个真实障碍物和对应远场数据，
然后分别对无噪声/有噪声数据计算正交采样指标函数，并把指标图保存出来。

参数向量沿用联合 GN 脚本的约定：每个障碍物 7 个参数
    [center_x, center_y, radius, a2c, a2s, a3c, a3s]
其中后四项是星形边界的 Fourier 形状扰动系数。
"""
from __future__ import annotations

# 标准库：命令行参数、JSON 元数据输出、路径处理。
import argparse
import json
from pathlib import Path

import matplotlib

# Agg 后端只负责生成图片文件，不打开 GUI 窗口，适合批量运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from common.scattering import PI2, Array, CArray, parse_float_list
from common.targets import plot_obstacle_boundaries
from common.direct_sampling import compute_direct_sampling_result
from common.reconstruction import build_true_params
from common.sampling import direct_sampling_indicator, plot_indicator_image


def save_imaging_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, p_true: Array, title: str) -> None:
    """保存直接成像指标图，并把真实障碍物边界叠加在图上。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)

    im = plot_indicator_image(ax, image, x_grid, y_grid, title=title)
    plot_obstacle_boundaries(ax, p_true, 3, "k--", lw=1.2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized indicator")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """主程序：生成真实数据、加噪声、计算直接成像指标并保存结果。"""
    p = argparse.ArgumentParser(
        description="Direct imaging for three small obstacles using multi-direction orthogonality sampling"
    )

    # ---------- 命令行参数 ----------
    p.add_argument("--out-dir", type=str, default="outputs_obstacle_direct_imaging")
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--radius", type=float, default=0.045)
    p.add_argument("--spacing", type=float, default=0.18)
    p.add_argument("--noise-levels", type=str, default="0.05,0.10,0.20")
    p.add_argument("--noise-level", type=float, default=None, help="deprecated single-noise override")
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
    p.add_argument("--block-size", type=int, default=32768)
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

    # 创建输出目录。
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 实验网格和方向离散 ----------
    k = float(args.k)
    noise_levels = np.asarray([float(args.noise_level)], dtype=float) if args.noise_level is not None else parse_float_list(args.noise_levels)
    incident_angles = parse_float_list(args.incident_angles)

    # 观测方向均匀覆盖单位圆。
    obs_angles = np.linspace(0.0, PI2, int(args.n_obs), endpoint=False)

    # 成像区域为 [-grid_extent, grid_extent]^2。
    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))

    # ---------- 生成远场数据 ----------
    p_true, centers_true = build_true_params(args)

    # ---------- 生成远场数据，并计算无噪声/有噪声直接成像指标 ----------
    result = compute_direct_sampling_result(
        p_true,
        3,
        int(args.n_per_obstacle),
        k,
        incident_angles,
        obs_angles,
        x_grid,
        y_grid,
        PI2,
        noise_levels,
        int(args.seed) + 999,
        indicator_power=float(args.indicator_power),
        block_size=int(args.block_size),
        noise_seed_stride=1,
    )
    image_clean = result.image_clean

    # ---------- 保存图片和数据 ----------
    save_imaging_plot(
        out_dir / "direct_imaging_clean.png",
        image_clean,
        x_grid,
        y_grid,
        p_true,
        title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), clean data",
    )
    noisy_plot_paths = []
    for noise_level, image_noisy in zip(noise_levels, result.image_noisy_list):
        noisy_plot = out_dir / f"direct_imaging_noisy_{float(noise_level):.2f}.png"
        save_imaging_plot(
            noisy_plot,
            image_noisy,
            x_grid,
            y_grid,
            p_true,
            title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), noise={float(noise_level):.2f}",
        )
        noisy_plot_paths.append(str(noisy_plot))

    # npz 文件保存所有核心数组，便于后续不重跑前向问题直接分析。
    np.savez_compressed(
        out_dir / "direct_imaging_result.npz",
        p_true=p_true,
        centers_true=centers_true,
        farfield_clean=result.farfield_clean,
        farfield_noisy=np.stack(result.farfield_noisy_list, axis=0),
        image_clean=image_clean,
        image_noisy=np.stack(result.image_noisy_list, axis=0),
        x_grid=x_grid,
        y_grid=y_grid,
        obs_angles=obs_angles,
        incident_angles=incident_angles,
        k=k,
        noise_levels=noise_levels,
    )

    # metadata.json 保存实验说明和主要输出路径。
    metadata = {
        "method": "multi-direction orthogonality sampling",
        "indicator_formula": "mu_MD(y,k)=sum_d |sum_xhat exp(i*k*(xhat dot y)) u_inf(xhat,d,k)|",
        "indicator_power": float(args.indicator_power),
        "block_size": int(args.block_size),
        "colormap": "jet",
        "k": k,
        "noise_levels": noise_levels.tolist(),
        "seed": int(args.seed),
        "centers_true": centers_true.tolist(),
        "output_clean_plot": str(out_dir / "direct_imaging_clean.png"),
        "output_noisy_plots": noisy_plot_paths,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 运行结束时在终端打印摘要。
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    # 直接作为脚本运行时执行 main；作为模块导入时不会自动跑实验。
    main()
