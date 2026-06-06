#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared obstacle reconstruction algorithms.

This module contains reusable quantitative-reconstruction routines that were
originally embedded in ``obstacle_joint_gn.py`` and are now shared by several
experiment scripts (prior-sensitivity, hybrid imaging, direct imaging).

All functions here operate on the seven-parameter star-shaped obstacle
parameterisation used throughout the project.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve, svd

from common.scattering import Array, CArray, direction_vectors
from common.sampling import normalize_indicator
from common.targets import deduplicate_legend, obstacle_param_slice, plot_obstacle_boundaries
from common.forward import solve_forward_farfield


# ---------------------------------------------------------------------------
# 中心参数工具
# ---------------------------------------------------------------------------

def centers_from_params(params: Array) -> Array:
    """Extract obstacle centers from a concatenated 21-element parameter vector."""
    return np.array(
        [
            [params[obstacle_param_slice(j).start], params[obstacle_param_slice(j).start + 1]]
            for j in range(3)
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# MUSIC 指标
# ---------------------------------------------------------------------------

def music_indicator(
    farfield_matrix: CArray,
    k: float,
    obs_angles: Array,
    x_grid: Array,
    y_grid: Array,
    rank_signal: int,
    block_size: int = 32768,
) -> Array:
    """Compute a MUSIC pseudo-spectrum image from far-field data.

    The first ``rank_signal`` singular vectors span the signal subspace; the
    remainder span the noise subspace.  At a true scatterer location the test
    vector is nearly orthogonal to the noise subspace, giving a large value of
    ``1 / ||P_noise phi(y)||``.
    """
    U, _, _ = svd(farfield_matrix, full_matrices=False)
    rank = max(1, min(int(rank_signal), U.shape[1]))
    U_noise = U[:, rank:]
    xhat = direction_vectors(obs_angles)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    denom = np.empty(pts.shape[0], dtype=float)
    if U_noise.size == 0:
        denom.fill(1e-12)
    else:
        scale = np.sqrt(len(obs_angles))
        U_noise_h = U_noise.conj().T
        for start in range(0, pts.shape[0], block_size):
            stop = min(start + block_size, pts.shape[0])
            phase = np.exp(-1j * k * (xhat @ pts[start:stop].T)) / scale
            proj = U_noise_h @ phase
            denom[start:stop] = np.linalg.norm(proj, axis=0)
    ind = 1.0 / (denom + 1e-12)
    ind = ind.reshape(X.shape)
    return normalize_indicator(ind)


# ---------------------------------------------------------------------------
# 峰值选择
# ---------------------------------------------------------------------------

def select_peaks_2d(
    image: Array,
    x_grid: Array,
    y_grid: Array,
    n_peaks: int,
    exclusion_radius: float,
) -> Array:
    """Pick ``n_peaks`` local maxima from a 2-D image with an exclusion radius."""
    img = image.copy()
    centers: List[Array] = []
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    for _ in range(n_peaks):
        idx = np.unravel_index(np.argmax(img), img.shape)
        centers.append(np.array([X[idx], Y[idx]], dtype=float))
        mask = (X - X[idx]) ** 2 + (Y - Y[idx]) ** 2 <= exclusion_radius ** 2
        img[mask] = -np.inf
    return np.vstack(centers)


# ---------------------------------------------------------------------------
# 约束与分离
# ---------------------------------------------------------------------------

def obstacle_max_radius(coeffs: Array) -> float:
    """Estimate the largest possible radius of a star-shaped obstacle."""
    r0 = float(coeffs[0])
    return r0 * (1.0 + np.sum(np.abs(coeffs[1:])))


def enforce_constraints(
    params: Array,
    min_gap: float,
    radius_bounds: Tuple[float, float],
    coeff_bounds: Tuple[float, float],
    center_extent: float,
) -> Array:
    """Project obstacle parameters into the feasible set.

    Constraints: centers inside [-center_extent, center_extent]; radii and
    Fourier coefficients within given bounds; obstacles separated by at least
    ``min_gap``.
    """
    p = params.copy()
    for j in range(3):
        sl = obstacle_param_slice(j)
        p[sl.start] = np.clip(p[sl.start], -center_extent, center_extent)
        p[sl.start + 1] = np.clip(p[sl.start + 1], -center_extent, center_extent)
        p[sl.start + 2] = np.clip(p[sl.start + 2], radius_bounds[0], radius_bounds[1])
        for idx in range(sl.start + 3, sl.stop):
            p[idx] = np.clip(p[idx], coeff_bounds[0], coeff_bounds[1])
    for _ in range(6):
        moved = False
        centers = np.array(
            [
                [p[obstacle_param_slice(j).start], p[obstacle_param_slice(j).start + 1]]
                for j in range(3)
            ],
            dtype=float,
        )
        req_extra = [
            obstacle_max_radius(p[obstacle_param_slice(j).start + 2 : obstacle_param_slice(j).stop])
            for j in range(3)
        ]
        for i, j in itertools.combinations(range(3), 2):
            dvec = centers[j] - centers[i]
            d = np.linalg.norm(dvec)
            req = req_extra[i] + req_extra[j] + min_gap
            if d < req:
                direction = dvec / d if d > 1e-12 else np.array([1.0, 0.0])
                mid = 0.5 * (centers[i] + centers[j])
                half = 0.5 * req
                centers[i] = mid - half * direction
                centers[j] = mid + half * direction
                moved = True
        for j in range(3):
            p[obstacle_param_slice(j).start] = np.clip(centers[j, 0], -center_extent, center_extent)
            p[obstacle_param_slice(j).start + 1] = np.clip(centers[j, 1], -center_extent, center_extent)
        if not moved:
            break
    return p


# ---------------------------------------------------------------------------
# Gauss-Newton 重建
# ---------------------------------------------------------------------------

def gauss_newton_reconstruct(
    farfield_noisy: CArray,
    init_params: Array,
    k: float,
    n_per_obstacle: int,
    incident_angles: Array,
    obs_angles: Array,
    n_iter: int,
    lambda_reg: float,
    damping: float,
    radius_bounds: Tuple[float, float],
    coeff_bounds: Tuple[float, float],
    min_gap: float,
    center_extent: float,
) -> Tuple[Array, List[Dict[str, Any]]]:
    """Joint damped Gauss-Newton refinement of three star-shaped obstacles.

    At each iteration the predicted far-field is compared with the noisy
    observation; a regularised normal equation is solved via a central-
    difference Jacobian.
    """
    params = enforce_constraints(init_params, min_gap, radius_bounds, coeff_bounds, center_extent)
    history: List[Dict[str, Any]] = []

    pscale = np.tile(
        np.array([1.0, 1.0, 0.3, 0.18, 0.18, 0.14, 0.14], dtype=float), 3
    )

    def flatten(z: CArray) -> Array:
        return np.concatenate([np.real(z).ravel(), np.imag(z).ravel()])

    target_vec = flatten(farfield_noisy)
    clip_template = np.tile(
        np.array([0.03, 0.03, 0.012, 0.025, 0.025, 0.02, 0.02], dtype=float), 3
    )

    for it in range(n_iter):
        ff = solve_forward_farfield(params, k, n_per_obstacle, incident_angles, obs_angles)
        resid = target_vec - flatten(ff)
        m = resid.size
        npar = len(params)
        J = np.empty((m, npar), dtype=float)
        for ell in range(npar):
            h = 1e-3 * max(abs(params[ell]), 1.0)
            p_plus = params.copy()
            p_plus[ell] += h
            p_minus = params.copy()
            p_minus[ell] -= h
            p_plus = enforce_constraints(p_plus, min_gap, radius_bounds, coeff_bounds, center_extent)
            p_minus = enforce_constraints(p_minus, min_gap, radius_bounds, coeff_bounds, center_extent)
            f_plus = flatten(
                solve_forward_farfield(p_plus, k, n_per_obstacle, incident_angles, obs_angles)
            )
            f_minus = flatten(
                solve_forward_farfield(p_minus, k, n_per_obstacle, incident_angles, obs_angles)
            )
            J[:, ell] = (f_plus - f_minus) / (2.0 * h)

        reg = lambda_reg * np.diag(1.0 / (pscale ** 2))
        delta = solve(J.T @ J + reg, J.T @ resid, assume_a="pos")
        delta = np.clip(delta, -clip_template, clip_template)
        params = params + damping * delta
        params = enforce_constraints(params, min_gap, radius_bounds, coeff_bounds, center_extent)

        centers = np.array(
            [
                [params[obstacle_param_slice(j).start], params[obstacle_param_slice(j).start + 1]]
                for j in range(3)
            ],
            dtype=float,
        )
        rel_res = float(np.linalg.norm(resid) / max(np.linalg.norm(target_vec), 1e-14))
        history.append(
            {
                "iteration": it + 1,
                "centers": centers.tolist(),
                "relative_residual": rel_res,
            }
        )
    return params, history


# ---------------------------------------------------------------------------
# 中心距离与分辨判断
# ---------------------------------------------------------------------------

def pairwise_min_distance(centers: Array) -> float:
    """Minimum pairwise distance among a set of center points."""
    return float(
        min(
            np.linalg.norm(centers[j] - centers[i])
            for i, j in itertools.combinations(range(len(centers)), 2)
        )
    )


def best_center_match_error(true_centers: Array, est_centers: Array) -> Tuple[float, float]:
    """Optimal permutation matching — returns (mean_error, max_error)."""
    best_mean = float("inf")
    best_max = float("inf")
    for perm in itertools.permutations(range(len(est_centers))):
        diffs = np.linalg.norm(est_centers[list(perm)] - true_centers, axis=1)
        mean_err = float(np.mean(diffs))
        max_err = float(np.max(diffs))
        if mean_err < best_mean:
            best_mean, best_max = mean_err, max_err
    return best_mean, best_max


def resolved_from_centers(
    true_centers: Array, rec_centers: Array, true_spacing: float
) -> Tuple[bool, float, float]:
    """Decide whether three obstacles are resolved based on center errors."""
    mean_err, max_err = best_center_match_error(true_centers, rec_centers)
    rec_dmin = pairwise_min_distance(rec_centers)
    tol = max(0.07, 0.35 * true_spacing)
    ok = bool(max_err <= tol and rec_dmin >= 0.6 * true_spacing)
    return ok, mean_err, max_err


# ---------------------------------------------------------------------------
# 随机中心生成
# ---------------------------------------------------------------------------

def generate_random_centers(
    spacing: float,
    rng: np.random.Generator,
    extent: float,
    min_pair_gap: float,
    max_tries: int = 5000,
) -> Array:
    """Generate three irregular centers satisfying spacing and extent constraints."""
    centers: List[Array] = []
    target_min = spacing
    for _ in range(max_tries):
        cand = rng.uniform(-extent, extent, size=2)
        if all(np.linalg.norm(cand - c) >= target_min for c in centers):
            centers.append(cand)
            if len(centers) == 3:
                break
    if len(centers) < 3:
        raise RuntimeError(
            "failed to generate random irregular centers with the requested spacing"
        )
    arr = np.vstack(centers)
    for _ in range(20):
        centroid = np.mean(arr, axis=0)
        arr = arr - 0.15 * centroid[None, :]
        ok = True
        for i, j in itertools.combinations(range(3), 2):
            if np.linalg.norm(arr[j] - arr[i]) < target_min:
                ok = False
                break
        if ok and np.max(np.abs(arr)) <= extent:
            break
    if np.std(arr[:, 1]) < 0.03:
        arr[:, 1] += rng.uniform(-0.05, 0.05, size=3)
    for _ in range(20):
        moved = False
        for i, j in itertools.combinations(range(3), 2):
            dvec = arr[j] - arr[i]
            d = np.linalg.norm(dvec)
            req = target_min + min_pair_gap
            if d < req:
                direction = dvec / d if d > 1e-12 else np.array([1.0, 0.0])
                mid = 0.5 * (arr[i] + arr[j])
                half = 0.5 * req
                arr[i] = np.clip(mid - half * direction, -extent, extent)
                arr[j] = np.clip(mid + half * direction, -extent, extent)
                moved = True
        if not moved:
            break
    return arr


# ---------------------------------------------------------------------------
# 从命令行参数构造真实障碍物参数
# ---------------------------------------------------------------------------

def build_true_params(args: object) -> tuple[Array, Array]:
    """Build a 21-element true-parameter vector and (3,2) center array.

    Reads ``center_extent``, ``seed``, ``spacing``, ``min_gap``, ``radius``,
    and ``true{1,2,3}_a{2,3}{c,s}`` from the argument namespace.
    """
    center_extent = float(getattr(args, "center_extent"))
    rng_cent = np.random.default_rng(int(getattr(args, "seed")))
    centers_true = generate_random_centers(
        float(getattr(args, "spacing")), rng_cent, center_extent,
        float(getattr(args, "min_gap")),
    )
    coeffs_true = [
        np.array(
            [
                float(getattr(args, "radius")),
                float(getattr(args, f"true{j}_a2c")),
                float(getattr(args, f"true{j}_a2s")),
                float(getattr(args, f"true{j}_a3c")),
                float(getattr(args, f"true{j}_a3s")),
            ],
            dtype=float,
        )
        for j in range(1, 4)
    ]
    p_true = np.concatenate(
        [np.concatenate([centers_true[j], coeffs_true[j]]) for j in range(3)]
    ).astype(float)
    return p_true, centers_true


# ---------------------------------------------------------------------------
# 绘图辅助
# ---------------------------------------------------------------------------

def save_gn_case_plot(
    path: Path,
    p_true: Array,
    p_init: Array,
    p_rec: Array,
    title: str,
) -> None:
    """Save a boundary comparison plot for one GN reconstruction case."""
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    for p, style, label in [
        (p_true, "k--", "true"),
        (p_init, "b:", "init"),
        (p_rec, "r-", "reconstructed"),
    ]:
        plot_obstacle_boundaries(ax, p, 3, style, lw=1.5, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    deduplicate_legend(ax)
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
