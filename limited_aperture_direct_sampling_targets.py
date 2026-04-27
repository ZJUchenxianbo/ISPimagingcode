#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""有限孔径条件下的直接采样/正交采样成像实验。

脚本比较三类目标：
1. 一个小目标；
2. 三个小目标；
3. 一个较大目标。

与全孔径观测不同，这里观测方向只覆盖一个角度区间
    [aperture_center - alpha, aperture_center + alpha]
因此成像会出现方向性模糊或分辨率下降。脚本会分别保存无噪声和有噪声指标图。
"""
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

# 复用三小障碍物 GN 脚本中的边界离散、单层势矩阵、远场算子等基础工具。
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
    """一个待成像目标案例的配置。

    name/label 用于文件名和图标题；
    params 是拼接后的障碍物参数向量；
    n_obstacles 是障碍物个数；
    n_boundary 是每个障碍物边界离散点数；
    grid_extent 控制该案例的成像区域大小。
    """
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
    """构造一个星形障碍物的 7 维参数块。

    参数格式：
        [center_x, center_y, radius, a2c, a2s, a3c, a3s]
    """
    return np.array([center[0], center[1], radius, a2c, a2s, a3c, a3s], dtype=float)


def make_cases() -> list[TargetCase]:
    """生成脚本中要比较的三个目标案例。"""
    # 单个小目标：位置接近原点，半径很小，形状有轻微非圆扰动。
    one_small = np.concatenate([
        make_star_block((0.02, -0.01), 0.045, 0.10, -0.06, 0.04, 0.02),
    ])

    # 三个小目标：用于观察有限孔径下多目标分辨能力。
    three_small = np.concatenate([
        make_star_block((-0.16, -0.08), 0.043, 0.10, -0.05, 0.04, 0.02),
        make_star_block((0.12, -0.02), 0.046, -0.08, 0.08, -0.04, 0.03),
        make_star_block((0.00, 0.15), 0.044, 0.06, 0.09, 0.04, -0.05),
    ])

    # 单个大目标：用来对比尺寸变大后指标图对边界/区域的响应。
    one_large = np.concatenate([
        make_star_block((0.02, 0.00), 0.18, 0.22, -0.08, 0.07, 0.05),
    ])
    return [
        TargetCase("one_small", "1 small target", one_small, 1, 80, 0.45),
        TargetCase("three_small", "3 small targets", three_small, 3, 48, 0.50),
        TargetCase("one_large", "1 large target", one_large, 1, 180, 0.60),
    ]


def limited_aperture_angles(center: float, alpha: float, n_obs: int) -> Array:
    """生成有限孔径观测角。

    center 是孔径中心方向，alpha 是半孔径角。
    当 alpha=pi 时退化为全孔径，即 [0, 2pi) 上均匀观测。
    """
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
    """支持任意障碍物个数的远场前向求解。

    three_small_obstacles_joint_gn_random_centers.solve_forward_farfield 默认固定 3 个障碍物；
    这里为了同时处理 1 个和 3 个目标，单独包装一个 n_obstacles 可变的版本。
    """
    # 根据参数向量离散所有障碍物边界。
    geom: BoundaryGeometry = params_to_geometry(params, n_per_obstacle, n_obstacles=n_obstacles)

    # 单层势边界积分方程矩阵，以及把边界密度映射到远场的算子。
    A = build_single_layer_matrix(geom, k)
    Ainf = single_layer_farfield_operator(geom, k, obs_angles)

    # 很小的对角稳定项，缓解矩阵病态带来的数值问题。
    eye = np.eye(A.shape[0], dtype=complex)
    stab = 1e-12 * max(np.linalg.norm(A, ord=2), 1.0)
    farfield = np.empty((len(obs_angles), len(incident_angles)), dtype=complex)
    for j, ang in enumerate(incident_angles):
        # 入射平面波方向 d=(cos ang, sin ang)。
        d = np.array([math.cos(float(ang)), math.sin(float(ang))], dtype=float)

        # 声软障碍物边界条件 u_total=0，因此单层势密度方程右端为 -u_inc。
        rhs = -plane_wave(geom.x, k, d)
        density = solve(A + stab * eye, rhs, assume_a="gen")

        # 由边界密度计算各观测方向上的远场模式。
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
    """计算有限孔径多方向直接采样指标。

    与全孔径正交采样的区别在于：
    观测方向积分只在有限孔径区间上进行，因此观测权重使用 aperture_length。
    对每个采样点 y，先把远场数据按 exp(i*k*xhat·y) 反传播，
    再对入射方向求和，得到归一化指标图。
    """
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])

    # 成像网格展平成点列表，便于矩阵化计算相位项。
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # 有限孔径上的梯形/矩形求积近似权重。
    obs_weight = aperture_length / max(len(obs_angles) - 1, 1)
    inc_weight = PI2 / len(incident_angles)

    # phase[m,n] = exp(i*k*xhat_m · y_n)。
    phase = np.exp(1j * k * (xhat @ pts.T))

    # 对观测方向做反传播积分。
    backpropagated = obs_weight * (farfield_matrix.T @ phase)

    # 对多个入射方向累加，得到最终采样指标。
    indicator = inc_weight * np.sum(np.abs(backpropagated) ** float(power), axis=0)
    indicator = indicator.reshape(X.shape).astype(float, copy=False)

    # 归一化到最大值为 1，方便统一颜色条。
    indicator /= max(float(np.max(indicator)), 1e-14)
    return indicator


def boundary_sets(case: TargetCase) -> list[Array]:
    """返回某个案例中每个障碍物的密集边界点，用于画真实轮廓。"""
    return [
        dense_boundary_points(case.params[obstacle_param_slice(j)], n=500)
        for j in range(case.n_obstacles)
    ]


def save_case_plot(path: Path, image: Array, x_grid: Array, y_grid: Array, case: TargetCase, title: str) -> None:
    """保存单个案例的指标图，并叠加真实目标边界。"""
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
    """把多个目标案例并排画在同一张总览图中。"""
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
    """主程序：对每个目标案例生成有限孔径远场数据并计算直接采样指标。"""
    p = argparse.ArgumentParser(
        description="Fixed-frequency limited-aperture direct sampling images for small and large sound-soft targets."
    )

    # ---------- 命令行参数 ----------
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
    p.add_argument("--noise-levels", type=str, default="0.05,0.10,0.20")
    p.add_argument("--noise-level", type=float, default=None, help="deprecated single-noise override")
    p.add_argument("--indicator-power", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()

    # 创建输出目录。
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 观测孔径、入射方向和随机数 ----------
    k = float(args.k)
    alpha = float(args.alpha)

    # alpha=pi 表示全孔径，孔径长度为 2pi；否则有限孔径长度为 2alpha。
    aperture_length = PI2 if abs(alpha - math.pi) < 1e-14 else 2.0 * alpha
    obs_angles = limited_aperture_angles(float(args.aperture_center), alpha, int(args.n_obs))
    incident_angles = parse_float_list(args.incident_angles)
    noise_levels = np.asarray([float(args.noise_level)], dtype=float) if args.noise_level is not None else parse_float_list(args.noise_levels)

    cases = make_cases()
    clean_images: list[Array] = []
    noisy_images_by_noise: dict[float, list[Array]] = {float(noise): [] for noise in noise_levels}
    grids: list[tuple[Array, Array]] = []
    metadata_cases = []

    for case in cases:
        # 每个案例可以有不同成像范围，因此单独生成网格。
        x_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        y_grid = np.linspace(-case.grid_extent, case.grid_extent, int(args.grid_size))
        grids.append((x_grid, y_grid))

        # 先生成无噪声远场，再添加指定相对噪声。
        farfield_clean = solve_forward_farfield_variable(
            case.params,
            case.n_obstacles,
            k,
            case.n_boundary,
            incident_angles,
            obs_angles,
        )
        # 分别计算无噪声和有噪声指标图。
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
        clean_images.append(image_clean)

        # 保存单案例图片。
        save_case_plot(
            out_dir / f"{case.name}_clean.png",
            image_clean,
            x_grid,
            y_grid,
            case,
            f"{case.label}, clean data",
        )
        farfield_noisy_list = []
        image_noisy_list = []
        noisy_plot_paths = []
        for idx, noise_level in enumerate(noise_levels):
            rng = np.random.default_rng(int(args.seed) + 1000 * idx)
            farfield_noisy = add_relative_noise(farfield_clean, float(noise_level), rng)
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
            noisy_plot = out_dir / f"{case.name}_noisy_{float(noise_level):.2f}.png"
            save_case_plot(
                noisy_plot,
                image_noisy,
                x_grid,
                y_grid,
                case,
                f"{case.label}, noise={float(noise_level):.2f}",
            )
            farfield_noisy_list.append(farfield_noisy)
            image_noisy_list.append(image_noisy)
            noisy_images_by_noise[float(noise_level)].append(image_noisy)
            noisy_plot_paths.append(str(noisy_plot))

        # 保存该案例所有核心数组。
        np.savez_compressed(
            out_dir / f"{case.name}_result.npz",
            params=case.params,
            farfield_clean=farfield_clean,
            farfield_noisy=np.stack(farfield_noisy_list, axis=0),
            image_clean=image_clean,
            image_noisy=np.stack(image_noisy_list, axis=0),
            x_grid=x_grid,
            y_grid=y_grid,
            obs_angles=obs_angles,
            incident_angles=incident_angles,
            k=k,
            alpha=alpha,
            noise_levels=noise_levels,
            aperture_center=float(args.aperture_center),
        )

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
