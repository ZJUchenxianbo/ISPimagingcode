#!/usr/bin/env python3
"""
Fixed-frequency joint Gauss-Newton reconstruction for three small sound-soft obstacles in 2D
with random irregular center locations.

Each obstacle is parameterized separately by a low-order star-like Fourier boundary around its own center:

    r(theta) = r0 * (1 + a2c*cos(2 theta) + a2s*sin(2 theta)
                        + a3c*cos(3 theta) + a3s*sin(3 theta)).

The script generates synthetic far-field data with a Nyström-style single-layer boundary integral solver,
adds prescribed noise levels, initializes three centers from a coarse MUSIC image, and then refines
all obstacle parameters jointly by a damped Gauss-Newton iteration.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import solve, svd
from scipy.special import hankel1

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
PI2 = 2.0 * np.pi


@dataclass
class BoundaryGeometry:
    x: Array
    normal: Array
    ds: Array
    obs_id: NDArray[np.int64]


@dataclass
class CaseMetrics:
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


def parse_float_list(text: str) -> Array:
    vals = [float(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return np.asarray(vals, dtype=float)


def star_radius(theta: Array, r0: float, a2c: float, a2s: float, a3c: float, a3s: float) -> Array:
    return r0 * (
        1.0
        + a2c * np.cos(2.0 * theta)
        + a2s * np.sin(2.0 * theta)
        + a3c * np.cos(3.0 * theta)
        + a3s * np.sin(3.0 * theta)
    )


def star_radius_derivative(theta: Array, r0: float, a2c: float, a2s: float, a3c: float, a3s: float) -> Array:
    return r0 * (
        -2.0 * a2c * np.sin(2.0 * theta)
        + 2.0 * a2s * np.cos(2.0 * theta)
        - 3.0 * a3c * np.sin(3.0 * theta)
        + 3.0 * a3s * np.cos(3.0 * theta)
    )


def star_boundary(center: Tuple[float, float], coeffs: Array, n_pts: int) -> Tuple[Array, Array, Array]:
    r0, a2c, a2s, a3c, a3s = [float(v) for v in coeffs]
    t = np.linspace(0.0, PI2, n_pts, endpoint=False)
    r = star_radius(t, r0, a2c, a2s, a3c, a3s)
    rp = star_radius_derivative(t, r0, a2c, a2s, a3c, a3s)
    ct = np.cos(t)
    st = np.sin(t)
    x = np.column_stack([center[0] + r * ct, center[1] + r * st])
    dx = rp * ct - r * st
    dy = rp * st + r * ct
    speed = np.sqrt(dx * dx + dy * dy)
    ds = speed * (PI2 / n_pts)
    normal = np.column_stack([dy / speed, -dx / speed])
    return x, normal, ds


def obstacle_param_slice(j: int) -> slice:
    return slice(7 * j, 7 * (j + 1))


def params_to_geometry(params: Array, n_per_obstacle: int, n_obstacles: int = 3) -> BoundaryGeometry:
    xs: List[Array] = []
    normals: List[Array] = []
    dss: List[Array] = []
    ids: List[Array] = []
    for j in range(n_obstacles):
        block = params[obstacle_param_slice(j)]
        center = (float(block[0]), float(block[1]))
        coeffs = block[2:7]
        x, nrm, ds = star_boundary(center, coeffs, n_per_obstacle)
        xs.append(x)
        normals.append(nrm)
        dss.append(ds)
        ids.append(np.full(n_per_obstacle, j, dtype=int))
    return BoundaryGeometry(x=np.vstack(xs), normal=np.vstack(normals), ds=np.concatenate(dss), obs_id=np.concatenate(ids))


def dense_boundary_points(params_obs: Array, n: int = 400) -> Array:
    center = (float(params_obs[0]), float(params_obs[1]))
    return star_boundary(center, params_obs[2:7], n)[0]


def plane_wave(x: Array, k: float, d: Array) -> CArray:
    return np.exp(1j * k * (x @ d))


def _diag_single_layer_integral(k: float, h: float) -> complex:
    if h <= 0.0:
        return 0.0 + 0.0j

    def f_re(s: float) -> float:
        return float(np.real(0.25j * hankel1(0, k * s)))

    def f_im(s: float) -> float:
        return float(np.imag(0.25j * hankel1(0, k * s)))

    a, b = 0.0, 0.5 * h
    re_val = quad(f_re, a, b, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    im_val = quad(f_im, a, b, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    return re_val + 1j * im_val


def build_single_layer_matrix(geom: BoundaryGeometry, k: float) -> CArray:
    n = geom.x.shape[0]
    A = np.empty((n, n), dtype=complex)
    for i in range(n):
        diff = geom.x[i][None, :] - geom.x
        rho = np.linalg.norm(diff, axis=1)
        row = 0.25j * hankel1(0, k * rho) * geom.ds
        row[i] = _diag_single_layer_integral(k, float(geom.ds[i]))
        A[i, :] = row
    return A


def single_layer_farfield_operator(geom: BoundaryGeometry, k: float, obs_angles: Array) -> CArray:
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])
    const = np.exp(1j * np.pi / 4.0) / np.sqrt(8.0 * np.pi * k)
    phase = np.exp(-1j * k * (xhat @ geom.x.T))
    return const * phase * geom.ds[None, :]


def solve_forward_farfield(params: Array, k: float, n_per_obstacle: int, incident_angles: Array, obs_angles: Array) -> CArray:
    geom = params_to_geometry(params, n_per_obstacle, n_obstacles=3)
    A = build_single_layer_matrix(geom, k)
    Ainf = single_layer_farfield_operator(geom, k, obs_angles)
    eye = np.eye(A.shape[0], dtype=complex)
    stab = 1e-12 * max(np.linalg.norm(A, ord=2), 1.0)
    farfield = np.empty((len(obs_angles), len(incident_angles)), dtype=complex)
    for j, ang in enumerate(incident_angles):
        d = np.array([math.cos(float(ang)), math.sin(float(ang))], dtype=float)
        rhs = -plane_wave(geom.x, k, d)
        density = solve(A + stab * eye, rhs, assume_a="gen")
        farfield[:, j] = Ainf @ density
    return farfield


def add_relative_noise(data: CArray, rel_noise: float, rng: np.random.Generator) -> CArray:
    if rel_noise <= 0.0:
        return data.copy()
    noise = rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
    noise /= max(np.linalg.norm(noise), 1e-14)
    amp = rel_noise * np.linalg.norm(data)
    return data + amp * noise


def empirical_snr(clean: CArray, noisy: CArray) -> float:
    return float(np.linalg.norm(clean) / max(np.linalg.norm(noisy - clean), 1e-14))


def music_indicator(farfield_matrix: CArray, k: float, obs_angles: Array, x_grid: Array, y_grid: Array, rank_signal: int) -> Array:
    U, _, _ = svd(farfield_matrix, full_matrices=False)
    rank = max(1, min(int(rank_signal), U.shape[1]))
    U_noise = U[:, rank:]
    xhat = np.column_stack([np.cos(obs_angles), np.sin(obs_angles)])
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    phase = np.exp(-1j * k * (xhat @ pts.T)) / np.sqrt(len(obs_angles))
    if U_noise.size == 0:
        denom = np.full(pts.shape[0], 1e-12, dtype=float)
    else:
        proj = U_noise.conj().T @ phase
        denom = np.linalg.norm(proj, axis=0)
    ind = 1.0 / (denom + 1e-12)
    ind = ind.reshape(X.shape)
    ind /= np.max(ind)
    return ind


def select_peaks_2d(image: Array, x_grid: Array, y_grid: Array, n_peaks: int, exclusion_radius: float) -> Array:
    img = image.copy()
    centers: List[Array] = []
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    for _ in range(n_peaks):
        idx = np.unravel_index(np.argmax(img), img.shape)
        centers.append(np.array([X[idx], Y[idx]], dtype=float))
        mask = (X - X[idx]) ** 2 + (Y - Y[idx]) ** 2 <= exclusion_radius ** 2
        img[mask] = -np.inf
    return np.vstack(centers)


def obstacle_max_radius(coeffs: Array) -> float:
    r0 = float(coeffs[0])
    return r0 * (1.0 + np.sum(np.abs(coeffs[1:])))


def enforce_constraints(params: Array, min_gap: float, radius_bounds: Tuple[float, float], coeff_bounds: Tuple[float, float], center_extent: float) -> Array:
    p = params.copy()
    for j in range(3):
        sl = obstacle_param_slice(j)
        p[sl.start] = np.clip(p[sl.start], -center_extent, center_extent)
        p[sl.start + 1] = np.clip(p[sl.start + 1], -center_extent, center_extent)
        p[sl.start + 2] = np.clip(p[sl.start + 2], radius_bounds[0], radius_bounds[1])
        for idx in range(sl.start + 3, sl.stop):
            p[idx] = np.clip(p[idx], coeff_bounds[0], coeff_bounds[1])
    # pairwise separation
    for _ in range(6):
        moved = False
        centers = np.array([[p[obstacle_param_slice(j).start], p[obstacle_param_slice(j).start + 1]] for j in range(3)], dtype=float)
        req_extra = [obstacle_max_radius(p[obstacle_param_slice(j).start + 2:obstacle_param_slice(j).stop]) for j in range(3)]
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
) -> Tuple[Array, List[Dict[str, float]]]:
    params = enforce_constraints(init_params, min_gap, radius_bounds, coeff_bounds, center_extent)
    history: List[Dict[str, float]] = []
    pscale = np.tile(np.array([1.0, 1.0, 0.3, 0.18, 0.18, 0.14, 0.14], dtype=float), 3)

    def flatten(z: CArray) -> Array:
        return np.concatenate([np.real(z).ravel(), np.imag(z).ravel()])

    target_vec = flatten(farfield_noisy)
    clip_template = np.tile(np.array([0.03, 0.03, 0.012, 0.025, 0.025, 0.02, 0.02], dtype=float), 3)
    for it in range(n_iter):
        ff = solve_forward_farfield(params, k, n_per_obstacle, incident_angles, obs_angles)
        resid = target_vec - flatten(ff)
        m = resid.size
        npar = len(params)
        J = np.empty((m, npar), dtype=float)
        for ell in range(npar):
            h = 1e-3 * max(abs(params[ell]), 1.0)
            p_plus = params.copy(); p_plus[ell] += h
            p_minus = params.copy(); p_minus[ell] -= h
            p_plus = enforce_constraints(p_plus, min_gap, radius_bounds, coeff_bounds, center_extent)
            p_minus = enforce_constraints(p_minus, min_gap, radius_bounds, coeff_bounds, center_extent)
            f_plus = flatten(solve_forward_farfield(p_plus, k, n_per_obstacle, incident_angles, obs_angles))
            f_minus = flatten(solve_forward_farfield(p_minus, k, n_per_obstacle, incident_angles, obs_angles))
            J[:, ell] = (f_plus - f_minus) / (2.0 * h)
        reg = lambda_reg * np.diag(1.0 / (pscale ** 2))
        delta = solve(J.T @ J + reg, J.T @ resid, assume_a="pos")
        delta = np.clip(delta, -clip_template, clip_template)
        params = params + damping * delta
        params = enforce_constraints(params, min_gap, radius_bounds, coeff_bounds, center_extent)

        centers = np.array([[params[obstacle_param_slice(j).start], params[obstacle_param_slice(j).start + 1]] for j in range(3)], dtype=float)
        rel_res = float(np.linalg.norm(resid) / max(np.linalg.norm(target_vec), 1e-14))
        history.append({
            "iteration": it + 1,
            "centers": centers.tolist(),
            "relative_residual": rel_res,
        })
    return params, history


def save_case_plot(path: Path, p_true: Array, p_init: Array, p_rec: Array, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    for p, style, label in [(p_true, "k--", "true"), (p_init, "b:", "init"), (p_rec, "r-", "reconstructed")]:
        for j in range(3):
            pts = dense_boundary_points(p[obstacle_param_slice(j)])
            ax.plot(pts[:, 0], pts[:, 1], style, lw=1.5, label=label if j == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_panel(path: Path, true_params_by_spacing: List[Array], init_map: Dict[Tuple[int, int], Array], rec_map: Dict[Tuple[int, int], Array], spacings: Array, noises: Array, d_rayleigh: float) -> None:
    nrows, ncols = len(noises), len(spacings)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    if axes_arr.shape != (nrows, ncols):
        if nrows == 1:
            axes_arr = axes_arr.reshape(1, ncols)
        elif ncols == 1:
            axes_arr = axes_arr.reshape(nrows, 1)
    for i, noise in enumerate(noises):
        for j, spacing in enumerate(spacings):
            ax = axes_arr[i, j]
            for p, style in [(true_params_by_spacing[j], "k--"), (init_map[(j, i)], "b:"), (rec_map[(j, i)], "r-")]:
                for q in range(3):
                    pts = dense_boundary_points(p[obstacle_param_slice(q)])
                    ax.plot(pts[:, 0], pts[:, 1], style, lw=1.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"noise={noise:.2f}, d_min={spacing:.3f}\n d_min/d_R={spacing / d_rayleigh:.2f}")
    fig.suptitle("Three general star-like obstacles with random irregular centers; joint Gauss-Newton")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_resolution_curve(path: Path, metrics: List[CaseMetrics], noises: Array, d_rayleigh: float) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for noise in noises:
        rows = [m for m in metrics if abs(m.noise - noise) < 1e-12]
        rows.sort(key=lambda z: z.spacing_true_min)
        x = np.array([m.spacing_true_min / d_rayleigh for m in rows])
        y = np.array([1.0 if m.resolved else 0.0 for m in rows])
        ax.plot(x, y, marker="o", lw=2, label=f"noise={noise:.2f}")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, label=r"$d_R$")
    ax.set_xlabel(r"minimum center spacing / $d_R$")
    ax.set_ylabel("resolved (1) / unresolved (0)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Joint Gauss-Newton for three general star-like obstacles with irregular centers")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pairwise_min_distance(centers: Array) -> float:
    return float(min(np.linalg.norm(centers[j] - centers[i]) for i, j in itertools.combinations(range(len(centers)), 2)))


def best_center_match_error(true_centers: Array, est_centers: Array) -> Tuple[float, float]:
    best_mean = float("inf")
    best_max = float("inf")
    for perm in itertools.permutations(range(len(est_centers))):
        diffs = np.linalg.norm(est_centers[list(perm)] - true_centers, axis=1)
        mean_err = float(np.mean(diffs))
        max_err = float(np.max(diffs))
        if mean_err < best_mean:
            best_mean, best_max = mean_err, max_err
    return best_mean, best_max


def resolved_from_centers(true_centers: Array, rec_centers: Array, true_spacing: float) -> Tuple[bool, float, float]:
    mean_err, max_err = best_center_match_error(true_centers, rec_centers)
    rec_dmin = pairwise_min_distance(rec_centers)
    tol = max(0.07, 0.35 * true_spacing)
    ok = bool(max_err <= tol and rec_dmin >= 0.6 * true_spacing)
    return ok, mean_err, max_err


def generate_random_centers(spacing: float, rng: np.random.Generator, extent: float, min_pair_gap: float, max_tries: int = 5000) -> Array:
    centers: List[Array] = []
    target_min = spacing
    for _ in range(max_tries):
        cand = rng.uniform(-extent, extent, size=2)
        if all(np.linalg.norm(cand - c) >= target_min for c in centers):
            centers.append(cand)
            if len(centers) == 3:
                break
    if len(centers) < 3:
        raise RuntimeError("failed to generate random irregular centers with the requested spacing")
    arr = np.vstack(centers)
    # if random points are too spread, gently pull them toward the origin while keeping min spacing
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
    # add tiny random y-perturbation if nearly collinear on x-axis
    if np.std(arr[:, 1]) < 0.03:
        arr[:, 1] += rng.uniform(-0.05, 0.05, size=3)
    # final check and separation using repulsion if needed
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


def run_experiment(args: argparse.Namespace) -> Dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        rng_cent = np.random.default_rng(int(args.seed) + 100 * j)
        centers_true = generate_random_centers(float(spacing), rng_cent, center_extent, float(args.min_gap))
        p_true_blocks = []
        for q in range(3):
            p_true_blocks.append(np.concatenate([centers_true[q], coeffs_true[q]]))
        p_true = np.concatenate(p_true_blocks).astype(float)
        true_params_by_spacing.append(p_true.copy())
        ff_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)

        spacing_dir = out_dir / f"spacing_{j:02d}_{spacing:.4f}"
        spacing_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(spacing_dir / "farfield_clean.npz", farfield_clean=ff_clean, k=k, spacing=float(spacing), d_rayleigh=d_rayleigh, p_true=p_true, centers_true=centers_true)

        for i, noise in enumerate(noises):
            rng = np.random.default_rng(int(args.seed) + 1000 * j + i)
            ff_noisy = add_relative_noise(ff_clean, float(noise), rng)
            img = music_indicator(ff_noisy, k, obs_angles, x_grid, y_grid, rank_signal=3)
            centers_init = select_peaks_2d(img, x_grid, y_grid, n_peaks=3, exclusion_radius=max(0.08, 0.45 * float(spacing)))
            p_init_blocks = []
            for q in range(3):
                p_init_blocks.append(np.concatenate([centers_init[q], np.array([float(args.init_radius), 0.0, 0.0, 0.0, 0.0])]))
            p_init = np.concatenate(p_init_blocks).astype(float)
            p_init = enforce_constraints(p_init, float(args.min_gap), (float(args.min_radius), float(args.max_radius)), (float(args.min_coeff), float(args.max_coeff)), center_extent)
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
            ff_rec = solve_forward_farfield(p_rec, k, int(args.n_per_obstacle), incident_angles, obs_angles)
            rel_res = float(np.linalg.norm(ff_rec - ff_noisy) / max(np.linalg.norm(ff_noisy), 1e-14))

            centers_init_arr = np.array([[p_init[obstacle_param_slice(q).start], p_init[obstacle_param_slice(q).start + 1]] for q in range(3)], dtype=float)
            centers_rec_arr = np.array([[p_rec[obstacle_param_slice(q).start], p_rec[obstacle_param_slice(q).start + 1]] for q in range(3)], dtype=float)
            resolved, mean_err, max_err = resolved_from_centers(centers_true, centers_rec_arr, float(spacing))

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
    p = argparse.ArgumentParser(description="Three general star-like obstacles with random irregular centers: joint Gauss-Newton")
    p.add_argument("--out-dir", type=str, default="outputs_three_small_obstacles_joint_gn_random_centers")
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--radius", type=float, default=0.045)
    p.add_argument("--spacing-list", type=str, default="0.30,0.18")
    p.add_argument("--noise-levels", type=str, default="0.01,0.05,0.10")
    p.add_argument("--incident-angles", type=str, default="0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438")
    p.add_argument("--n-per-obstacle", type=int, default=10)
    p.add_argument("--n-obs", type=int, default=10)
    p.add_argument("--grid-extent", type=float, default=0.45)
    p.add_argument("--grid-size", type=int, default=41)
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
    parser = build_argparser()
    args = parser.parse_args()
    outputs = run_experiment(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
