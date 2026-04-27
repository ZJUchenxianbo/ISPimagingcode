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

# 从联合 Gauss-Newton 脚本复用几何、前向散射、噪声和绘图辅助函数。
from three_small_obstacles_joint_gn_random_centers import (
    PI2,
    add_relative_noise,
    dense_boundary_points,
    generate_random_centers,
    obstacle_param_slice,
    parse_float_list,
    solve_forward_farfield,
)

# 实数/复数 numpy 数组类型别名，让函数签名更清楚。
Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def build_true_params(args: argparse.Namespace) -> tuple[Array, Array]:
    """根据命令行参数构造三个真实障碍物的完整参数向量。

    真实中心由 generate_random_centers 随机产生，但受 seed 控制，因此可复现。
    每个障碍物共 7 个参数：中心 2 个、半径 1 个、形状 Fourier 系数 4 个。
    返回：
        p_true: 拼接后的完整参数向量，长度 21。
        centers_true: 三个真实中心坐标，形状为 (3, 2)。
    """
    center_extent = float(args.center_extent)
    rng_cent = np.random.default_rng(int(args.seed))

    # 生成三个彼此至少相隔 spacing、且不超出中心范围的随机中心。
    centers_true = generate_random_centers(float(args.spacing), rng_cent, center_extent, float(args.min_gap))

    # 三个障碍物的真实形状系数。radius 是基准半径，后四项控制边界非圆形扰动。
    coeffs_true = [
        np.array([float(args.radius), float(args.true1_a2c), float(args.true1_a2s), float(args.true1_a3c), float(args.true1_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true2_a2c), float(args.true2_a2s), float(args.true2_a3c), float(args.true2_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true3_a2c), float(args.true3_a2s), float(args.true3_a3c), float(args.true3_a3s)], dtype=float),
    ]

    # 每个障碍物块为 [center_x, center_y, radius, a2c, a2s, a3c, a3s]。
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
    """计算多入射方向的正交采样指标函数。

    farfield_matrix 的维度通常是：
        (观测方向数, 入射方向数)

    对每个采样点 y，代码近似计算
        sum_d | sum_xhat exp(i*k*xhat·y) u_inf(xhat,d) |^power
    再归一化到最大值为 1。指标值越大，说明该点越像散射体所在位置。
    """
    # 观测方向 xhat=(cos theta, sin theta)。
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])

    # 构造成像平面上的二维采样点，并展平成 (网格点数, 2)。
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # 用均匀求积权重近似单位圆上的积分。
    obs_weight = PI2 / len(obs_angles)
    inc_weight = PI2 / len(incident_angles)

    # phase[m, n] = exp(i*k*xhat_m · y_n)，用于把远场数据反传播到采样点。
    phase = np.exp(1j * k * (xhat @ pts.T))

    # 对观测方向积分；farfield_matrix.T 让每个入射方向独立得到一个反传播场。
    reduced_fields = obs_weight * (farfield_matrix.T @ phase)

    # 对所有入射方向累加，得到每个采样点的指标值。
    indicator = inc_weight * np.sum(np.abs(reduced_fields) ** float(power), axis=0)
    indicator = indicator.reshape(X.shape).astype(float, copy=False)

    # 归一化，便于不同噪声/实验之间比较颜色尺度。
    indicator /= max(np.max(indicator), 1e-14)
    return indicator


def save_imaging_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, p_true: Array, title: str) -> None:
    """保存直接成像指标图，并把真实障碍物边界叠加在图上。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)

    # pcolormesh 显示指标函数；颜色越亮，表示越可能存在障碍物。
    m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r")
    for j in range(3):
        # dense_boundary_points 把参数化边界采样成密集点，用虚线画真实轮廓。
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
    """主程序：生成真实数据、加噪声、计算直接成像指标并保存结果。"""
    p = argparse.ArgumentParser(
        description="Direct imaging for three small obstacles using multi-direction orthogonality sampling"
    )

    # ---------- 命令行参数 ----------
    p.add_argument("--out-dir", type=str, default="outputs_three_small_obstacles_direct_imaging")
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

    # 无噪声远场数据。
    farfield_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)

    # 使用固定 seed 的随机数生成器添加相对噪声。
    # ---------- 计算无噪声/有噪声直接成像指标 ----------
    image_clean = orthogonality_sampling_indicator_md(
        farfield_clean,
        k,
        obs_angles,
        incident_angles,
        x_grid,
        y_grid,
        power=float(args.indicator_power),
    )
    # ---------- 保存图片和数据 ----------
    save_imaging_plot(
        out_dir / "direct_imaging_clean.png",
        image_clean,
        x_grid,
        y_grid,
        p_true,
        title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), clean data",
    )
    farfield_noisy_list = []
    image_noisy_list = []
    noisy_plot_paths = []
    for idx, noise_level in enumerate(noise_levels):
        rng_noise = np.random.default_rng(int(args.seed) + 999 + idx)
        farfield_noisy = add_relative_noise(farfield_clean, float(noise_level), rng_noise)
        image_noisy = orthogonality_sampling_indicator_md(
            farfield_noisy,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            power=float(args.indicator_power),
        )
        noisy_plot = out_dir / f"direct_imaging_noisy_{float(noise_level):.2f}.png"
        save_imaging_plot(
            noisy_plot,
            image_noisy,
            x_grid,
            y_grid,
            p_true,
            title=f"Direct imaging (orthogonality sampling, p={args.indicator_power:g}), noise={float(noise_level):.2f}",
        )
        farfield_noisy_list.append(farfield_noisy)
        image_noisy_list.append(image_noisy)
        noisy_plot_paths.append(str(noisy_plot))

    # npz 文件保存所有核心数组，便于后续不重跑前向问题直接分析。
    np.savez_compressed(
        out_dir / "direct_imaging_result.npz",
        p_true=p_true,
        centers_true=centers_true,
        farfield_clean=farfield_clean,
        farfield_noisy=np.stack(farfield_noisy_list, axis=0),
        image_clean=image_clean,
        image_noisy=np.stack(image_noisy_list, axis=0),
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
        "colormap": "RdYlBu_r",
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
