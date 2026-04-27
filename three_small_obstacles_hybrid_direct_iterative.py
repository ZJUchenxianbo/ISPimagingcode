#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid qualitative-then-quantitative reconstruction for three small sound-soft obstacles.

Workflow:
1. Generate the same three-obstacle far-field data as the existing direct-imaging / GN scripts.
2. Use multi-direction orthogonality sampling for a qualitative direct image.
3. Extract prior information from the image (centers + rough radii).
4. Use the extracted prior as initialization for joint Gauss-Newton quantitative reconstruction.

中文说明：
这个脚本把两个重建思想串起来使用：
1. 先用正交采样法（orthogonality sampling）做“定性直接成像”，得到一个归一化指标图。
   指标图中亮的区域通常对应障碍物可能存在的位置。
2. 再从指标图中自动提取三个障碍物的粗略中心和半径，作为反演迭代的初值。
3. 最后用联合 Gauss-Newton 方法同时优化三个障碍物的几何参数，得到更精细的定量重建结果。

每个障碍物的参数块长度为 7：
    [center_x, center_y, radius, a2c, a2s, a3c, a3s]
其中 center_x/center_y 是中心坐标，radius 是基准半径，
a2c/a2s/a3c/a3s 是边界形状的二、三阶 Fourier 扰动系数。
"""
from __future__ import annotations

# 标准库：命令行参数、表格/JSON 输出、数学函数、路径处理、类型标注。
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# 使用 Agg 后端，表示只保存图片文件，不弹出交互式窗口；适合批量实验和服务器环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# 复用直接成像脚本中的“真实参数构造”和“多入射方向正交采样指标函数”。
from three_small_obstacles_direct_imaging import (
    build_true_params,
    orthogonality_sampling_indicator_md,
)

# 复用 Gauss-Newton 脚本里的前向求解、约束、峰值选择和误差评估工具。
from three_small_obstacles_joint_gn_random_centers import (
    PI2,
    add_relative_noise,
    dense_boundary_points,
    empirical_snr,
    enforce_constraints,
    gauss_newton_reconstruct,
    obstacle_param_slice,
    pairwise_min_distance,
    parse_float_list,
    resolved_from_centers,
    select_peaks_2d,
    solve_forward_farfield,
)

# 为 numpy 数组起别名，便于阅读函数签名。
# Array 表示实数数组，CArray 表示复数数组；本脚本中 CArray 只保留作类型约定。
Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def centers_from_params(params: Array) -> Array:
    """从完整参数向量中抽取三个障碍物的中心坐标。

    参数向量按障碍物分块排列，每块长度为 7：
        [x, y, r, a2c, a2s, a3c, a3s]
    这里仅取每个分块的前两个分量，即中心坐标 (x, y)。
    """
    return np.array(
        [
            # obstacle_param_slice(j) 返回第 j 个障碍物在完整参数向量中的切片。
            # .start 是该障碍物参数块的起始索引，因此 start 和 start+1 分别是 x/y。
            [params[obstacle_param_slice(j).start], params[obstacle_param_slice(j).start + 1]]
            for j in range(3)
        ],
        dtype=float,
    )


def estimate_prior_from_indicator(
    image: Array,
    x_grid: Array,
    y_grid: Array,
    n_targets: int,
    exclusion_radius: float,
    threshold_ratio: float,
    default_radius: float,
    radius_bounds: Tuple[float, float],
    radius_scale: float,
) -> Tuple[Array, Array]:
    """从直接成像指标图中估计障碍物中心和粗略半径。

    输入的 image 是已经归一化到 [0, 1] 附近的指标图。
    基本思路：
    1. 先在指标图里找 n_targets 个局部峰值，作为粗略中心。
    2. 取超过 threshold_ratio * max(image) 的高亮网格点，认为这些点属于障碍物支撑区域。
    3. 把高亮点分配给最近的粗略中心。
    4. 对每个目标，用局部高亮点的加权质心修正中心，并用高亮区域面积估计等效半径。

    注意：直接成像得到的是“位置先验”，不是精确边界；半径估计只用于给迭代法一个合理初值。
    """
    # 先通过非极大值抑制式的峰值搜索找到 n_targets 个候选中心。
    # exclusion_radius 用来避免在同一个亮斑附近重复选峰。
    coarse_centers = select_peaks_2d(
        image,
        x_grid,
        y_grid,
        n_peaks=n_targets,
        exclusion_radius=exclusion_radius,
    )

    # 网格间距用于把“高亮网格点数量”换算为面积。
    dx = float(abs(x_grid[1] - x_grid[0])) if len(x_grid) > 1 else 1.0
    dy = float(abs(y_grid[1] - y_grid[0])) if len(y_grid) > 1 else 1.0

    # X/Y 是二维网格坐标；pts 是展平后的所有采样点，形状为 (网格点数, 2)。
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    vals = image.ravel()

    # 只保留足够亮的点。1e-14 是防止图像全零时出现数值问题。
    mask = vals >= float(threshold_ratio) * max(float(np.max(image)), 1e-14)

    # 默认先使用峰值中心和给定默认半径；若后续提取失败，就退回这些稳妥值。
    refined_centers = coarse_centers.copy()
    radii = np.full(n_targets, float(default_radius), dtype=float)
    if not np.any(mask):
        # 没有任何点超过阈值时，说明指标图太弱或阈值太高，直接返回粗略峰值。
        return refined_centers, radii

    # support_pts/support_vals 分别是高亮点的位置和对应指标值。
    support_pts = pts[mask]
    support_vals = vals[mask]

    # 计算每个高亮点到每个粗略中心的距离，然后分配给最近的中心。
    # dists 的形状为 (高亮点数, n_targets)。
    dists = np.linalg.norm(support_pts[:, None, :] - coarse_centers[None, :, :], axis=2)
    owners = np.argmin(dists, axis=1)

    for j in range(n_targets):
        # 取出被分配给第 j 个目标的高亮点。
        local = owners == j
        if not np.any(local):
            # 如果某个峰值附近没有高亮支撑点，则保持默认中心和半径。
            continue
        local_pts = support_pts[local]
        local_vals = support_vals[local]
        wsum = float(np.sum(local_vals))
        if wsum > 1e-14:
            # 用指标值作为权重做质心，亮度越高的点对中心估计影响越大。
            refined_centers[j] = np.sum(local_pts * local_vals[:, None], axis=0) / wsum
        else:
            # 极端情况下权重和太小，则退化为普通几何平均。
            refined_centers[j] = np.mean(local_pts, axis=0)

        # 用局部高亮点数量估计区域面积，再换成等效圆半径：
        # area = pi * r^2 => r = sqrt(area / pi)。
        area_est = float(local_pts.shape[0]) * dx * dy
        equiv_radius = radius_scale * math.sqrt(max(area_est, 1e-14) / math.pi)

        # 直接面积估计可能偏大/偏小，因此和默认半径做 50%-50% 混合，提高稳定性。
        blended_radius = 0.5 * float(default_radius) + 0.5 * equiv_radius

        # 把半径限制在允许范围内，避免给 Gauss-Newton 一个不合理的初值。
        radii[j] = float(np.clip(blended_radius, radius_bounds[0], radius_bounds[1]))

    return refined_centers, radii


def build_init_params_from_prior(
    centers_prior: Array,
    radii_prior: Array,
    center_extent: float,
    min_gap: float,
    radius_bounds: Tuple[float, float],
    coeff_bounds: Tuple[float, float],
) -> Array:
    """把“直接成像提取的中心和半径”组装成 Gauss-Newton 初始参数向量。

    直接成像只能可靠给出中心和粗半径；形状扰动系数 a2c/a2s/a3c/a3s 暂时设为 0，
    即初始边界先看作圆。后续迭代会根据远场数据自动修正这些形状系数。
    """
    blocks = []
    for j in range(3):
        blocks.append(
            np.array(
                [
                    # 前两项：第 j 个障碍物中心坐标。
                    centers_prior[j, 0],
                    centers_prior[j, 1],
                    # 第三项：从指标图估计出的粗略半径。
                    radii_prior[j],
                    # 后四项：Fourier 形状扰动初值。设为 0 表示从圆形开始迭代。
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        )
    params = np.concatenate(blocks).astype(float)

    # enforce_constraints 会修正参数，使其满足：
    # 1. 中心坐标不超过 center_extent；
    # 2. 障碍物之间至少保留 min_gap；
    # 3. 半径和 Fourier 系数落在指定上下界内。
    return enforce_constraints(params, min_gap, radius_bounds, coeff_bounds, center_extent)


def save_direct_prior_plot(
    path: Path,
    image: Array,
    x_grid: Array,
    y_grid: Array,
    p_true: Array,
    p_init: Array,
    title: str,
) -> None:
    """保存“直接成像指标图 + 真实边界 + 初始先验边界”的对比图。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)

    # pcolormesh 显示归一化指标图：亮/暖色区域表示直接成像认为目标更可能存在。
    m = ax.pcolormesh(x_grid, y_grid, image, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    for j in range(3):
        # 将参数化边界离散成较密的点，便于画出障碍物轮廓。
        pts_true = dense_boundary_points(p_true[obstacle_param_slice(j)])
        pts_init = dense_boundary_points(p_init[obstacle_param_slice(j)])

        # 黑色虚线：真实障碍物；白色点线：由直接成像提取出的先验初值。
        # label 只在第一个障碍物上设置，避免图例重复三次。
        ax.plot(pts_true[:, 0], pts_true[:, 1], "k--", lw=1.2, label="true" if j == 0 else None)
        ax.plot(pts_init[:, 0], pts_init[:, 1], "w:", lw=1.4, label="prior from direct imaging" if j == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)

    # 去除重复 label，得到更干净的图例。
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("normalized indicator")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reconstruction_plot(path: Path, p_true: Array, p_init: Array, p_rec: Array, title: str) -> None:
    """保存真实边界、直接成像初值和最终迭代结果的三方对比图。"""
    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)

    # 三种曲线：
    # true：真实边界；
    # direct-imaging prior：直接成像提取出的初始边界；
    # iterative reconstruction：Gauss-Newton 迭代后的重建边界。
    for params, style, label in [
        (p_true, "k--", "true"),
        (p_init, "b:", "direct-imaging prior"),
        (p_rec, "r-", "iterative reconstruction"),
    ]:
        for j in range(3):
            pts = dense_boundary_points(params[obstacle_param_slice(j)])
            ax.plot(pts[:, 0], pts[:, 1], style, lw=1.4, label=label if j == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.18)

    # 合并重复图例项。
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary_panels(
    path: Path,
    images: List[Array],
    x_grid: Array,
    y_grid: Array,
    p_true: Array,
    init_params_list: List[Array],
    rec_params_list: List[Array],
    noise_levels: Array,
) -> None:
    """把所有噪声水平下的结果汇总成一张 2 行多列的面板图。

    第一行：直接成像指标图，以及由图像提取出的先验边界。
    第二行：真实边界、先验边界、Gauss-Newton 最终重建边界的对比。
    每一列对应一个噪声水平。
    """
    fig, axes = plt.subplots(2, len(noise_levels), figsize=(5.2 * len(noise_levels), 8.6), constrained_layout=True)

    # 当噪声水平只有一个时，matplotlib 可能返回一维/标量 axes；atleast_2d 统一成二维索引。
    axes = np.atleast_2d(axes)
    for col, noise in enumerate(noise_levels):
        # 上排：定性直接成像结果。
        ax_img = axes[0, col]
        m = ax_img.pcolormesh(x_grid, y_grid, images[col], shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
        for j in range(3):
            pts_true = dense_boundary_points(p_true[obstacle_param_slice(j)])
            pts_init = dense_boundary_points(init_params_list[col][obstacle_param_slice(j)])
            ax_img.plot(pts_true[:, 0], pts_true[:, 1], "k--", lw=1.0)
            ax_img.plot(pts_init[:, 0], pts_init[:, 1], "w:", lw=1.2)
        ax_img.set_aspect("equal")
        ax_img.set_title(f"Direct imaging prior, noise={noise:.2f}")
        ax_img.set_xlabel("x")
        ax_img.set_ylabel("y")
        ax_img.grid(True, alpha=0.15)

        # 下排：定量迭代结果对比。
        ax_rec = axes[1, col]
        for params, style in [
            (p_true, "k--"),
            (init_params_list[col], "b:"),
            (rec_params_list[col], "r-"),
        ]:
            for j in range(3):
                pts = dense_boundary_points(params[obstacle_param_slice(j)])
                ax_rec.plot(pts[:, 0], pts[:, 1], style, lw=1.2)
        ax_rec.set_aspect("equal")
        ax_rec.set_title(f"Quantitative iteration, noise={noise:.2f}")
        ax_rec.set_xlabel("x")
        ax_rec.set_ylabel("y")
        ax_rec.grid(True, alpha=0.15)

    # 对整张上排图共享同一个色条，表示归一化指标值。
    cbar = fig.colorbar(m, ax=axes[0, :].ravel().tolist(), shrink=0.85)
    cbar.set_label("normalized indicator")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """主程序：解析参数、生成数据、直接成像、迭代重建、保存结果。"""
    p = argparse.ArgumentParser(
        description="Use direct imaging to get prior information, then use joint Gauss-Newton for quantitative reconstruction."
    )

    # ---------- 实验和物理参数 ----------
    # out-dir：所有图片、npz、json、csv 结果的输出目录。
    p.add_argument("--out-dir", type=str, default="outputs_three_small_obstacles_hybrid_direct_iterative")

    # k：波数。Rayleigh 长度在本脚本中取 pi/k，用于报告障碍物间距相对分辨率。
    p.add_argument("--k", type=float, default=8.0)

    # radius：构造真实障碍物时的基准半径，同时也作为直接成像半径估计失败时的默认半径。
    p.add_argument("--radius", type=float, default=0.045)

    # spacing：随机生成三个中心时使用的目标间距尺度。
    p.add_argument("--spacing", type=float, default=0.18)

    # noise-levels：多个相对噪声水平，逗号分隔；脚本会逐个噪声水平做完整实验。
    p.add_argument("--noise-levels", type=str, default="0.05,0.10,0.20")

    # incident-angles：入射方向角列表，默认 8 个方向均匀覆盖 [0, 2pi)。
    p.add_argument(
        "--incident-angles",
        type=str,
        default="0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438",
    )

    # n-per-obstacle：每个障碍物边界离散点数，用于前向散射求解。
    p.add_argument("--n-per-obstacle", type=int, default=10)

    # n-obs：远场观测方向数量。
    p.add_argument("--n-obs", type=int, default=72)

    # grid-extent/grid-size：直接成像采样区域 [-extent, extent]^2 及每个方向网格点数。
    p.add_argument("--grid-extent", type=float, default=0.45)
    p.add_argument("--grid-size", type=int, default=161)

    # center-extent：障碍物中心坐标允许范围，即中心会被限制在 [-center_extent, center_extent]^2。
    p.add_argument("--center-extent", type=float, default=0.22)

    # min-gap：障碍物之间允许的最小间隔，避免重叠或过分接近导致前向/反演不稳定。
    p.add_argument("--min-gap", type=float, default=0.008)

    # 半径和 Fourier 形状系数的约束范围，用于初值修正和迭代过程投影。
    p.add_argument("--min-radius", type=float, default=0.03)
    p.add_argument("--max-radius", type=float, default=0.07)
    p.add_argument("--min-coeff", type=float, default=-0.18)
    p.add_argument("--max-coeff", type=float, default=0.18)

    # indicator-power：正交采样指标中 |reduced field| 的幂次。
    p.add_argument("--indicator-power", type=float, default=1.0)

    # prior-threshold：提取先验时，只使用超过该比例峰值的高亮区域。
    p.add_argument("--prior-threshold", type=float, default=0.60)

    # prior-radius-scale：由高亮区域面积换算半径时的缩放因子。
    p.add_argument("--prior-radius-scale", type=float, default=0.72)

    # Gauss-Newton 迭代参数：
    # lambda-reg 是 Tikhonov/阻尼型正则强度，damping 是每步更新的步长缩放，n-iter 是迭代次数。
    p.add_argument("--lambda-reg", type=float, default=1.0e-2)
    p.add_argument("--damping", type=float, default=0.7)
    p.add_argument("--n-iter", type=int, default=3)

    # 三个真实障碍物的 Fourier 形状扰动系数。
    # true1/true2/true3 分别对应三个障碍物；a2c/a2s/a3c/a3s 表示二、三阶 cos/sin 系数。
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

    # seed：随机种子，控制真实中心生成和噪声生成，保证实验可复现。
    p.add_argument("--seed", type=int, default=24680)
    args = p.parse_args()

    # 创建输出目录。如果目录已存在，不报错，后续文件会覆盖同名结果。
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 将命令行参数整理成数值对象 ----------
    k = float(args.k)

    # 这里把 Rayleigh 分辨尺度定义为 pi/k，用于判断真实障碍物间距是否低于/接近分辨极限。
    d_rayleigh = math.pi / k

    # 把逗号分隔字符串解析成 numpy 数组。
    noise_levels = parse_float_list(args.noise_levels)
    incident_angles = parse_float_list(args.incident_angles)

    # 观测方向均匀分布在单位圆上；endpoint=False 避免 0 和 2pi 重复。
    obs_angles = np.linspace(0.0, PI2, int(args.n_obs), endpoint=False)

    # 直接成像的二维采样网格。
    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))

    # 约束参数统一成 tuple，传给约束投影和 Gauss-Newton 过程。
    center_extent = float(args.center_extent)
    radius_bounds = (float(args.min_radius), float(args.max_radius))
    coeff_bounds = (float(args.min_coeff), float(args.max_coeff))

    # ---------- 构造真实模型并生成无噪声远场数据 ----------
    # p_true 是完整真实参数向量，centers_true 是三个真实中心。
    p_true, centers_true = build_true_params(args)

    # true_spacing 是三个真实中心之间的最小距离。
    true_spacing = pairwise_min_distance(centers_true)

    # solve_forward_farfield 根据真实几何参数计算远场矩阵。
    # 通常形状为 (观测方向数, 入射方向数)，元素为复数远场值。
    farfield_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)

    # 保存无噪声基准数据，便于之后单独分析或复现实验。
    np.savez_compressed(
        out_dir / "farfield_clean.npz",
        farfield_clean=farfield_clean,
        p_true=p_true,
        centers_true=centers_true,
        obs_angles=obs_angles,
        incident_angles=incident_angles,
        x_grid=x_grid,
        y_grid=y_grid,
        d_rayleigh=d_rayleigh,
    )

    # rows 用于最终写 summary.csv/json；
    # 三个 list 用于最后绘制跨噪声水平的汇总图。
    rows: List[Dict[str, object]] = []
    prior_images: List[Array] = []
    init_params_list: List[Array] = []
    rec_params_list: List[Array] = []

    # ---------- 对每个噪声水平分别做：加噪声 -> 直接成像 -> 提取先验 -> GN 迭代 ----------
    for idx, noise in enumerate(noise_levels):
        # 每个噪声水平使用不同但可复现的随机数种子。
        rng_noise = np.random.default_rng(int(args.seed) + 1000 + idx)

        # 按相对噪声水平给无噪声远场数据添加复噪声。
        farfield_noisy = add_relative_noise(farfield_clean, float(noise), rng_noise)

        # 多方向正交采样指标函数。输出 image 为二维归一化指标图。
        image = orthogonality_sampling_indicator_md(
            farfield_noisy,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            power=float(args.indicator_power),
        )
        prior_images.append(image)

        # 峰值排除半径：防止一个障碍物的同一亮斑贡献多个峰。
        # 取 max(0.08, 0.45*spacing) 是一个经验稳健设置。
        exclusion_radius = max(0.08, 0.45 * float(args.spacing))

        # 从指标图中提取三个中心和粗略半径。
        centers_prior, radii_prior = estimate_prior_from_indicator(
            image,
            x_grid,
            y_grid,
            n_targets=3,
            exclusion_radius=exclusion_radius,
            threshold_ratio=float(args.prior_threshold),
            default_radius=float(args.radius),
            radius_bounds=radius_bounds,
            radius_scale=float(args.prior_radius_scale),
        )

        # 把中心/半径先验组装成完整参数向量，作为 Gauss-Newton 初值。
        p_init = build_init_params_from_prior(
            centers_prior,
            radii_prior,
            center_extent=center_extent,
            min_gap=float(args.min_gap),
            radius_bounds=radius_bounds,
            coeff_bounds=coeff_bounds,
        )
        init_params_list.append(p_init.copy())

        # 以直接成像先验为初值，联合优化三个障碍物的中心、半径和形状扰动系数。
        # history 记录每次迭代的残差、步长等诊断信息。
        p_rec, history = gauss_newton_reconstruct(
            farfield_noisy,
            p_init,
            k=k,
            n_per_obstacle=int(args.n_per_obstacle),
            incident_angles=incident_angles,
            obs_angles=obs_angles,
            n_iter=int(args.n_iter),
            lambda_reg=float(args.lambda_reg),
            damping=float(args.damping),
            radius_bounds=radius_bounds,
            coeff_bounds=coeff_bounds,
            min_gap=float(args.min_gap),
            center_extent=center_extent,
        )
        rec_params_list.append(p_rec.copy())

        # ---------- 计算重建质量指标 ----------
        # 抽取初值和重建结果的中心坐标。
        centers_init = centers_from_params(p_init)
        centers_rec = centers_from_params(p_rec)

        # resolved_from_centers 会把估计中心和真实中心做匹配，并给出是否分辨成功、平均/最大中心误差。
        init_resolved, init_mean_err, init_max_err = resolved_from_centers(centers_true, centers_init, true_spacing)
        rec_resolved, rec_mean_err, rec_max_err = resolved_from_centers(centers_true, centers_rec, true_spacing)

        # 用最终重建参数重新解一次前向问题，看它和带噪观测数据的相对残差。
        farfield_rec = solve_forward_farfield(p_rec, k, int(args.n_per_obstacle), incident_angles, obs_angles)
        rel_residual = float(np.linalg.norm(farfield_rec - farfield_noisy) / max(np.linalg.norm(farfield_noisy), 1e-14))

        # 每个噪声水平单独建一个子目录，存放图片、数据和指标。
        noise_dir = out_dir / f"noise_{float(noise):.2f}"
        noise_dir.mkdir(parents=True, exist_ok=True)

        # 保存直接成像先验图。
        save_direct_prior_plot(
            noise_dir / "direct_prior.png",
            image,
            x_grid,
            y_grid,
            p_true,
            p_init,
            title=f"Qualitative direct imaging prior, noise={float(noise):.2f}",
        )

        # 保存迭代重建对比图。
        save_reconstruction_plot(
            noise_dir / "iterative_reconstruction.png",
            p_true,
            p_init,
            p_rec,
            title=f"Direct-imaging prior + iterative reconstruction, noise={float(noise):.2f}",
        )

        # 保存该噪声水平下的核心数组数据，后续可以不用重跑实验直接加载分析。
        np.savez_compressed(
            noise_dir / "result.npz",
            p_true=p_true,
            p_init=p_init,
            p_rec=p_rec,
            centers_true=centers_true,
            centers_init=centers_init,
            centers_rec=centers_rec,
            radii_prior=radii_prior,
            image=image,
            farfield_noisy=farfield_noisy,
            farfield_rec=farfield_rec,
            x_grid=x_grid,
            y_grid=y_grid,
            noise=float(noise),
        )

        # 保存 Gauss-Newton 每步迭代诊断信息。
        with open(noise_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # 汇总该噪声水平的标量/列表指标。
        row: Dict[str, object] = {
            "noise": float(noise),
            "d_rayleigh": d_rayleigh,
            "spacing_true_min": true_spacing,
            "spacing_true_over_dR": true_spacing / d_rayleigh,
            "snr_eff_nominal": 1.0 / float(noise),
            "snr_eff_empirical": empirical_snr(farfield_clean, farfield_noisy),
            "prior_centers": centers_init.tolist(),
            "prior_radii": radii_prior.tolist(),
            "prior_spacing_min": pairwise_min_distance(centers_init),
            "prior_resolved": bool(init_resolved),
            "prior_mean_center_error": init_mean_err,
            "prior_max_center_error": init_max_err,
            "rec_centers": centers_rec.tolist(),
            "rec_spacing_min": pairwise_min_distance(centers_rec),
            "rec_resolved": bool(rec_resolved),
            "rec_mean_center_error": rec_mean_err,
            "rec_max_center_error": rec_max_err,
            "rel_farfield_residual": rel_residual,
        }
        rows.append(row)

        # 每个噪声目录也保存一份 metrics.json，方便单独查看。
        with open(noise_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    # 所有噪声水平结束后，保存一张总览面板图。
    save_summary_panels(
        out_dir / "summary_panel.png",
        prior_images,
        x_grid,
        y_grid,
        p_true,
        init_params_list,
        rec_params_list,
        noise_levels,
    )

    # 保存 CSV 表格，便于用 Excel/Origin/Matlab 等工具继续处理。
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 保存完整元数据，包括方法说明、主要参数、输出路径和每个噪声水平的指标。
    metadata = {
        "method": "direct imaging prior + joint Gauss-Newton iteration",
        "direct_imaging_indicator": "mu_MD(y,k)=sum_d |sum_xhat exp(i*k*(xhat dot y)) u_inf(xhat,d,k)|",
        "iterative_method": "joint Gauss-Newton with finite-difference Jacobian",
        "noise_levels": noise_levels.tolist(),
        "k": k,
        "grid_size": int(args.grid_size),
        "n_obs": int(args.n_obs),
        "n_per_obstacle": int(args.n_per_obstacle),
        "prior_threshold": float(args.prior_threshold),
        "prior_radius_scale": float(args.prior_radius_scale),
        "summary_panel": str(out_dir / "summary_panel.png"),
        "summary_csv": str(out_dir / "summary.csv"),
        "rows": rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 在终端打印 summary.json 的内容，方便脚本运行结束时直接看到结果摘要。
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    # 作为脚本直接运行时进入 main；被其他脚本 import 时不会自动执行实验。
    main()
