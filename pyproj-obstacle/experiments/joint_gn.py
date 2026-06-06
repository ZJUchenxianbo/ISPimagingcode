#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-frequency joint Gauss-Newton reconstruction for three small sound-soft obstacles in 2D
with random irregular center locations.

Each obstacle is parameterized separately by a low-order star-like Fourier boundary around its own center:

    r(theta) = r0 * (1 + a2c*cos(2 theta) + a2s*sin(2 theta)
                        + a3c*cos(3 theta) + a3s*sin(3 theta)).

The script generates synthetic far-field data with a Nyström-style single-layer boundary integral solver,
adds prescribed noise levels, initializes three centers from a coarse MUSIC image, and then refines
all obstacle parameters jointly by a damped Gauss-Newton iteration.

中文说明：
这个文件是三小障碍物重建实验的核心模块。它负责：
1. 用星形 Fourier 边界参数化声软障碍物；
2. 用单层势边界积分方程生成合成远场数据；
3. 用 MUSIC 指标图为三个中心提供粗初值；
4. 用有限差分 Jacobian 的阻尼 Gauss-Newton 方法联合优化中心、半径和形状系数。

每个障碍物的参数块长度为 7：
    [center_x, center_y, radius, a2c, a2s, a3c, a3s]
三个障碍物拼接后，完整参数向量长度为 21。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

# 使用 Agg 后端：脚本只保存图片，不打开交互式窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve, svd
from common.scattering import PI2, Array, CArray, add_relative_noise, direction_vectors, empirical_snr, parse_float_list
from common.targets import (
    BoundaryGeometry,
    deduplicate_legend,
    obstacle_param_slice,
    params_to_geometry,
    plot_obstacle_boundaries,
    star_boundary,
    star_radius,
    star_radius_derivative,
)
from common.forward import (
    build_single_layer_matrix,
    plane_wave,
    single_layer_farfield_operator,
    solve_forward_farfield,
)
from common.sampling import normalize_indicator
from common.reconstruction import (
    best_center_match_error,
    enforce_constraints,
    gauss_newton_reconstruct,
    generate_random_centers,
    music_indicator,
    obstacle_max_radius,
    pairwise_min_distance,
    resolved_from_centers,
    select_peaks_2d,
)


@dataclass
class CaseMetrics:
    """单个实验案例的评价指标，后续写入 CSV/JSON。"""
    spacing_true_min: float
    spacing_init_min: float
    spacing_rec_min: float
    noise: float
    d_rayleigh: float
    spacing_over_dR: float
    srf_eff: float
    snr_eff_nominal: float
    snr_eff_empirical: float
    mean_center_error: float
    max_center_error: float
    resolved: bool
    rel_farfield_residual: float
    true_centers: List[List[float]]
    init_centers: List[List[float]]
    rec_centers: List[List[float]]

def run_experiment(args: argparse.Namespace) -> Dict[str, str]:
    """运行完整批量实验：多个真实间距 × 多个噪声水平。"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 参数整理 ----------
    k = float(args.k)
    base_radius = float(args.radius)
    spacings = parse_float_list(args.spacing_list)
    noises = parse_float_list(args.noise_levels)
    incident_angles = parse_float_list(args.incident_angles)
    obs_angles = np.linspace(0.0, PI2, int(args.n_obs), endpoint=False)
    d_rayleigh = math.pi / k
    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    center_extent = float(args.center_extent)

    # 三个真实障碍物的半径和形状系数。
    coeffs_true = [
        np.array([base_radius, float(args.true1_a2c), float(args.true1_a2s), float(args.true1_a3c), float(args.true1_a3s)], dtype=float),
        np.array([base_radius, float(args.true2_a2c), float(args.true2_a2s), float(args.true2_a3c), float(args.true2_a3s)], dtype=float),
        np.array([base_radius, float(args.true3_a2c), float(args.true3_a2s), float(args.true3_a3c), float(args.true3_a3s)], dtype=float),
    ]

    all_metrics: List[CaseMetrics] = []
    true_params_by_spacing: List[Array] = []
    init_map: Dict[Tuple[int, int], Array] = {}
    rec_map: Dict[Tuple[int, int], Array] = {}

    for j, spacing in enumerate(spacings):
        # 每个 spacing 生成一组三障碍物真实中心。
        rng_cent = np.random.default_rng(int(args.seed) + 100 * j)
        centers_true = generate_random_centers(float(spacing), rng_cent, center_extent, float(args.min_gap))

        # 拼接真实参数向量。
        p_true_blocks = []
        for q in range(3):
            p_true_blocks.append(np.concatenate([centers_true[q], coeffs_true[q]]))
        p_true = np.concatenate(p_true_blocks).astype(float)
        true_params_by_spacing.append(p_true.copy())

        # 生成无噪声远场数据，后续同一 spacing 下所有噪声水平共用。
        ff_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)

        spacing_dir = out_dir / f"spacing_{j:02d}_{spacing:.4f}"
        spacing_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(spacing_dir / "farfield_clean.npz", farfield_clean=ff_clean, k=k, spacing=float(spacing), d_rayleigh=d_rayleigh, p_true=p_true, centers_true=centers_true)

        for i, noise in enumerate(noises):
            # 添加噪声并用 MUSIC 指标图寻找中心初值。
            rng = np.random.default_rng(int(args.seed) + 1000 * j + i)
            ff_noisy = add_relative_noise(ff_clean, float(noise), rng)
            img = music_indicator(ff_noisy, k, obs_angles, x_grid, y_grid, rank_signal=3, block_size=int(args.music_block_size))
            centers_init = select_peaks_2d(img, x_grid, y_grid, n_peaks=3, exclusion_radius=max(0.08, 0.45 * float(spacing)))

            # MUSIC 只给中心；半径用 init_radius，形状扰动从 0 开始。
            p_init_blocks = []
            for q in range(3):
                p_init_blocks.append(np.concatenate([centers_init[q], np.array([float(args.init_radius), 0.0, 0.0, 0.0, 0.0])]))
            p_init = np.concatenate(p_init_blocks).astype(float)
            p_init = enforce_constraints(p_init, float(args.min_gap), (float(args.min_radius), float(args.max_radius)), (float(args.min_coeff), float(args.max_coeff)), center_extent)

            # 联合 Gauss-Newton 迭代。
            p_rec, history = gauss_newton_reconstruct(
                ff_noisy,
                p_init,
                k=k,
                n_per_obstacle=int(args.n_per_obstacle),
                incident_angles=incident_angles,
                obs_angles=obs_angles,
                n_iter=int(args.n_iter),
                lambda_reg=float(args.lambda_reg),
                damping=float(args.damping),
                radius_bounds=(float(args.min_radius), float(args.max_radius)),
                coeff_bounds=(float(args.min_coeff), float(args.max_coeff)),
                min_gap=float(args.min_gap),
                center_extent=center_extent,
            )
            init_map[(j, i)] = p_init.copy()
            rec_map[(j, i)] = p_rec.copy()

            # 用最终参数重新计算远场，评估与带噪数据的相对残差。
            ff_rec = solve_forward_farfield(p_rec, k, int(args.n_per_obstacle), incident_angles, obs_angles)
            rel_res = float(np.linalg.norm(ff_rec - ff_noisy) / max(np.linalg.norm(ff_noisy), 1e-14))

            # 提取中心并判断重建是否成功分辨三个目标。
            centers_init_arr = np.array([[p_init[obstacle_param_slice(q).start], p_init[obstacle_param_slice(q).start + 1]] for q in range(3)], dtype=float)
            centers_rec_arr = np.array([[p_rec[obstacle_param_slice(q).start], p_rec[obstacle_param_slice(q).start + 1]] for q in range(3)], dtype=float)
            resolved, mean_err, max_err = resolved_from_centers(centers_true, centers_rec_arr, float(spacing))

            # 汇总该实验组合的指标。
            metric = CaseMetrics(
                spacing_true_min=pairwise_min_distance(centers_true),
                spacing_init_min=pairwise_min_distance(centers_init_arr),
                spacing_rec_min=pairwise_min_distance(centers_rec_arr),
                noise=float(noise),
                d_rayleigh=d_rayleigh,
                spacing_over_dR=float(pairwise_min_distance(centers_true) / d_rayleigh),
                srf_eff=float(d_rayleigh / pairwise_min_distance(centers_true)),
                snr_eff_nominal=float(1.0 / noise),
                snr_eff_empirical=empirical_snr(ff_clean, ff_noisy),
                mean_center_error=mean_err,
                max_center_error=max_err,
                resolved=bool(resolved),
                rel_farfield_residual=rel_res,
                true_centers=centers_true.tolist(),
                init_centers=centers_init_arr.tolist(),
                rec_centers=centers_rec_arr.tolist(),
            )
            all_metrics.append(metric)

            # 保存当前 spacing/noise 的数组、指标、迭代历史和图片。
            noise_dir = spacing_dir / f"noise_{noise:.2f}"
            noise_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(noise_dir / "farfield_noisy.npz", farfield_noisy=ff_noisy, farfield_clean=ff_clean)
            np.savez_compressed(noise_dir / "music_image.npz", image=img, x_grid=x_grid, y_grid=y_grid)
            np.savez_compressed(noise_dir / "reconstruction_result.npz", p_true=p_true, p_init=p_init, p_rec=p_rec, centers_true=centers_true)
            with open(noise_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(asdict(metric), f, indent=2)
            with open(noise_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            save_case_plot(
                noise_dir / "reconstruction.png",
                p_true, p_init, p_rec,
                title=(f"d_min={metric.spacing_true_min:.3f} ({metric.spacing_over_dR:.2f} d_R), noise={noise:.2f}\n"
                       f"resolved={resolved}, d_rec_min={metric.spacing_rec_min:.3f}, err_max={metric.max_center_error:.3f}")
            )

    if not all_metrics:
        raise RuntimeError("no metrics generated")

    # ---------- 汇总输出 ----------
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_metrics[0]).keys()))
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))
    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "k": k,
            "radius": base_radius,
            "true_shape_coefficients": [c.tolist() for c in coeffs_true],
            "d_rayleigh": d_rayleigh,
            "requested_spacings": spacings.tolist(),
            "noise_levels": noises.tolist(),
            "music_block_size": int(args.music_block_size),
            "metrics": [asdict(m) for m in all_metrics],
        }, f, indent=2)

    save_panel(out_dir / "reconstruction_panel.png", true_params_by_spacing, init_map, rec_map, spacings, noises, d_rayleigh)
    save_resolution_curve(out_dir / "resolution_vs_spacing.png", all_metrics, noises, d_rayleigh)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "reconstruction_panel": str(out_dir / "reconstruction_panel.png"),
        "resolution_vs_spacing": str(out_dir / "resolution_vs_spacing.png"),
    }


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    p = argparse.ArgumentParser(description="Three general star-like obstacles with random irregular centers: joint Gauss-Newton")
    p.add_argument("--out-dir", type=str, default="outputs_obstacle_joint_gn")
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--radius", type=float, default=0.045)
    p.add_argument("--spacing-list", type=str, default="0.30,0.18")
    p.add_argument("--noise-levels", type=str, default="0.05,0.10,0.20")
    p.add_argument("--incident-angles", type=str, default="0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438")
    p.add_argument("--n-per-obstacle", type=int, default=10)
    p.add_argument("--n-obs", type=int, default=10)
    p.add_argument("--grid-extent", type=float, default=0.45)
    p.add_argument("--grid-size", type=int, default=41)
    p.add_argument("--music-block-size", type=int, default=32768)
    p.add_argument("--center-extent", type=float, default=0.22)
    p.add_argument("--init-radius", type=float, default=0.05)
    p.add_argument("--min-radius", type=float, default=0.03)
    p.add_argument("--max-radius", type=float, default=0.07)
    p.add_argument("--min-coeff", type=float, default=-0.18)
    p.add_argument("--max-coeff", type=float, default=0.18)
    p.add_argument("--min-gap", type=float, default=0.008)
    p.add_argument("--n-iter", type=int, default=2)
    p.add_argument("--lambda-reg", type=float, default=1.0e-2)
    p.add_argument("--damping", type=float, default=0.7)
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
    return p


def main() -> None:
    """命令行入口。"""
    parser = build_argparser()
    args = parser.parse_args()
    outputs = run_experiment(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
