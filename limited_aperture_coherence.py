#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""有限孔径论文的核函数相干性与 Gram 条件数实验。

连续对象是归一化孔径因子

    C_alpha(h) = (2 alpha)^{-1} int_{theta0-alpha}^{theta0+alpha}
                 exp(i k xhat(theta).h) d theta.

对 coherent receiver/incident apertures，归一化 Gram entry 是
``C_R(h) * C_I(h)``。本脚本测试 point-spread width、product coherence、
Gram conditioning 和 aperture-center design。它故意不运行 PDE 正问题求解
和神经网络路径，只作为原始物理 measurement dictionary 的轻量 baseline。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

from scattering_common import PI2, Array, CArray


THREE_DB_LEVEL = 2.0 ** -0.5
SINGLE_TRANSVERSE_3DB = 0.2214732353
SINGLE_STATIONARY_3DB = 0.8689866207
DUAL_EQUAL_TRANSVERSE_3DB = 0.1594583493
DUAL_EQUAL_STATIONARY_3DB = 0.6210788031


@dataclass(frozen=True)
class ApertureArc:
    """Two-dimensional angular aperture.

    ``center`` and ``alpha`` are measured in radians.  The sign convention is
    ``+1`` for a receiver factor and ``-1`` for an incident factor.  The sign
    changes the complex Gram entry but not its modulus.
    """

    center: float
    alpha: float
    sign: float = 1.0


def _arc_angles(arc: ApertureArc, n_quad: int) -> Array:
    if n_quad < 16:
        raise ValueError("n_quad should be at least 16")
    if not (0.0 < arc.alpha <= math.pi):
        raise ValueError("arc.alpha must be in (0, pi]")
    if math.isclose(arc.alpha, math.pi, rel_tol=0.0, abs_tol=1e-14):
        return np.linspace(0.0, PI2, n_quad, endpoint=False)
    return arc.center + np.linspace(-arc.alpha, arc.alpha, n_quad)


def aperture_factor(
    k: float,
    radii: Array,
    beta: float,
    arc: ApertureArc,
    n_quad: int,
) -> CArray:
    """Evaluate one normalized finite-arc factor for one direction beta.

    Parameters
    ----------
    k:
        Wavenumber.
    radii:
        One-dimensional array of separation lengths R.
    beta:
        Direction angle of the separation vector h = R(cos beta, sin beta).
    arc:
        Aperture geometry and sign convention.
    n_quad:
        Number of quadrature directions on the aperture.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1:
        raise ValueError("radii must be one-dimensional")
    angles = _arc_angles(arc, n_quad)
    phase_slope = np.cos(angles - beta)
    values = np.exp(1j * float(arc.sign) * float(k) * radii[:, None] * phase_slope[None, :])
    if math.isclose(arc.alpha, math.pi, rel_tol=0.0, abs_tol=1e-14):
        return np.mean(values, axis=1)
    return np.trapezoid(values, angles, axis=1) / (2.0 * arc.alpha)


def product_factor(
    k: float,
    radii: Array,
    beta: float,
    receiver: ApertureArc,
    incident: ApertureArc | None,
    n_quad: int,
) -> CArray:
    """Evaluate receiver-only or coherent receiver-incident coherence factor."""
    receiver_factor = aperture_factor(k, radii, beta, receiver, n_quad)
    if incident is None:
        return receiver_factor
    incident_factor = aperture_factor(k, radii, beta, incident, n_quad)
    return receiver_factor * incident_factor


def first_crossing(radii: Array, values: Array, level: float) -> float:
    """Return the first radius where ``values`` falls below ``level``.

    The profile is oscillatory, so this is a first-crossing definition rather
    than a monotone root solve.
    """
    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    if radii.shape != values.shape:
        raise ValueError("radii and values must have the same shape")
    hits = np.flatnonzero(values <= level)
    if hits.size == 0:
        return float("nan")
    idx = int(hits[0])
    if idx == 0:
        return float(radii[0])
    r0, r1 = float(radii[idx - 1]), float(radii[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    if abs(v1 - v0) <= 1e-14:
        return r1
    weight = (level - v0) / (v1 - v0)
    return r0 + weight * (r1 - r0)


def sin_over_x(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-14
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def airy_j1_over_x(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-14
    out[mask] = 2.0 * j1(x[mask]) / x[mask]
    return out


def angular_distance_mod_pi(delta: float) -> float:
    """Smallest angular distance to a stationary direction, modulo pi."""
    return abs((float(delta) + 0.5 * math.pi) % math.pi - 0.5 * math.pi)


def one_factor_prediction(alpha: float, delta: float) -> float:
    """Small-aperture 3 dB prediction in units of wavelength."""
    s = abs(math.sin(delta))
    if s < 1e-10:
        return SINGLE_STATIONARY_3DB / (alpha * alpha)
    return SINGLE_TRANSVERSE_3DB / (alpha * s)


def estimate_profile_width(
    k: float,
    beta: float,
    receiver: ApertureArc,
    incident: ApertureArc | None,
    n_quad: int,
    n_r: int,
    r_max: float,
    level: float,
) -> float:
    radii = np.linspace(0.0, float(r_max), int(n_r))
    profile = np.abs(product_factor(k, radii, beta, receiver, incident, n_quad))
    return first_crossing(radii, profile, level)


def run_width_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alphas_deg = [10.0, 15.0, 20.0, 30.0, 45.0]
    deltas_deg = [0.0, 30.0, 45.0, 90.0]
    rows: list[tuple[float, float, float, float, float]] = []
    for alpha_deg in alphas_deg:
        alpha = math.radians(alpha_deg)
        receiver = ApertureArc(center=0.0, alpha=alpha, sign=1.0)
        for delta_deg in deltas_deg:
            beta = math.radians(delta_deg)
            prediction = one_factor_prediction(alpha, beta) * wavelength
            r_max = max(6.0 * wavelength, 1.6 * prediction)
            measured = estimate_profile_width(
                k,
                beta,
                receiver,
                None,
                int(args.n_quad),
                int(args.n_r),
                r_max,
                float(args.level),
            )
            ratio = (measured / prediction) if prediction > 0 else float("nan")
            rows.append((alpha_deg, delta_deg, measured / wavelength, prediction / wavelength, ratio))

    table = np.asarray(rows, dtype=float)
    npz_path = out_dir / "one_factor_widths.npz"
    np.savez_compressed(
        npz_path,
        alpha_deg=table[:, 0],
        delta_deg=table[:, 1],
        measured_r_over_lambda=table[:, 2],
        predicted_r_over_lambda=table[:, 3],
        ratio=table[:, 4],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for delta_deg in deltas_deg:
        subset = [row for row in rows if float(row[1]) == delta_deg]
        xs = np.asarray([float(row[0]) for row in subset], dtype=float)
        ys = np.asarray([float(row[2]) for row in subset], dtype=float)
        pred = np.asarray([float(row[3]) for row in subset], dtype=float)
        ax.loglog(xs, ys, "o-", label=fr"measured $\delta={delta_deg:g}^\circ$")
        ax.loglog(xs, pred, "--", color=ax.lines[-1].get_color(), alpha=0.65)
    ax.set_xlabel(r"aperture half-angle $\alpha$ (degrees)")
    ax.set_ylabel(r"first 3 dB radius $R/\lambda$")
    ax.set_title("One-aperture finite-arc width")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "one_factor_widths.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "one_factor_widths.png")}


def run_dual_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alpha = math.radians(float(args.dual_alpha_deg))
    gammas_deg = [0.0, 20.0, 40.0, 60.0, 90.0]
    beta_grid = np.linspace(-math.pi, math.pi, int(args.n_beta), endpoint=False)
    rows: list[tuple[float, float, float, float, float]] = []
    for gamma_deg in gammas_deg:
        receiver = ApertureArc(center=0.0, alpha=alpha, sign=1.0)
        incident = ApertureArc(center=math.radians(gamma_deg), alpha=alpha, sign=-1.0)
        widths = []
        for beta in beta_grid:
            # This bound covers both alpha^{-2} stationary and alpha^{-1} transverse regimes.
            r_max = max(
                8.0 * wavelength,
                1.8 * SINGLE_STATIONARY_3DB * wavelength / (alpha * alpha),
            )
            widths.append(
                estimate_profile_width(
                    k,
                    float(beta),
                    receiver,
                    incident,
                    int(args.n_quad),
                    int(args.n_r),
                    r_max,
                    float(args.level),
                )
            )
        widths_arr = np.asarray(widths, dtype=float)
        finite = np.isfinite(widths_arr)
        worst_idx = int(np.nanargmax(widths_arr)) if np.any(finite) else 0
        beta_worst = float(beta_grid[worst_idx])
        worst_width = float(widths_arr[worst_idx])
        gamma = math.radians(gamma_deg)
        if gamma <= 2.0 * alpha:
            bandwidth_guide = alpha * alpha
            width_guide = DUAL_EQUAL_STATIONARY_3DB * wavelength / bandwidth_guide
        else:
            bandwidth_guide = max(alpha * alpha, alpha * abs(math.sin(gamma / 2.0)))
            width_guide = DUAL_EQUAL_TRANSVERSE_3DB * wavelength / bandwidth_guide
        rows.append((gamma_deg, math.degrees(beta_worst), worst_width / wavelength, width_guide / wavelength, bandwidth_guide))

    table = np.asarray(rows, dtype=float)
    npz_path = out_dir / "dual_aperture_worst_widths.npz"
    np.savez_compressed(
        npz_path,
        gamma_deg=table[:, 0],
        worst_beta_deg=table[:, 1],
        worst_r_over_lambda=table[:, 2],
        regime_guide_r_over_lambda=table[:, 3],
        effective_bandwidth=table[:, 4],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    xs = np.asarray([float(row[0]) for row in rows], dtype=float)
    ys = np.asarray([float(row[2]) for row in rows], dtype=float)
    guide = np.asarray([float(row[3]) for row in rows], dtype=float)
    ax.plot(xs, ys, "o-", label="measured worst direction")
    ax.plot(xs, guide, "--", label="common-stationary / gap guide")
    ax.set_xlabel(r"center separation $\gamma$ (degrees)")
    ax.set_ylabel(r"worst first 3 dB radius $R/\lambda$")
    ax.set_title(r"Equal dual apertures, $\alpha={:.0f}^\circ$".format(float(args.dual_alpha_deg)))
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.savefig(out_dir / "dual_aperture_worst_widths.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "dual_aperture_worst_widths.png")}


def sector_label(beta: float, receiver: ApertureArc, incident: ApertureArc) -> str:
    receiver_stationary = angular_distance_mod_pi(beta - receiver.center) <= receiver.alpha
    incident_stationary = angular_distance_mod_pi(beta - incident.center) <= incident.alpha
    if receiver_stationary and incident_stationary:
        return "C"
    if incident_stationary:
        return "M_I"
    if receiver_stationary:
        return "M_R"
    return "G"


def run_sector_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    configs = [
        ("equal_gamma_0", 20.0, 20.0, 0.0),
        ("equal_gamma_30", 20.0, 20.0, 30.0),
        ("equal_gamma_40", 20.0, 20.0, 40.0),
        ("equal_gamma_70", 20.0, 20.0, 70.0),
        ("unequal_gamma_40", 30.0, 10.0, 40.0),
    ]
    beta_grid = np.linspace(-math.pi, math.pi, int(args.n_beta), endpoint=False)
    rows: list[tuple[str, float, float, float, float, str, float]] = []
    for label, alpha_r_deg, alpha_i_deg, gamma_deg in configs:
        alpha_r = math.radians(alpha_r_deg)
        alpha_i = math.radians(alpha_i_deg)
        receiver = ApertureArc(0.0, alpha_r, 1.0)
        incident = ApertureArc(math.radians(gamma_deg), alpha_i, -1.0)
        r_max = max(
            8.0 * wavelength,
            1.8 * SINGLE_STATIONARY_3DB * wavelength / (min(alpha_r, alpha_i) ** 2),
        )
        for beta in beta_grid:
            width = estimate_profile_width(
                k,
                float(beta),
                receiver,
                incident,
                int(args.n_quad),
                int(args.n_r),
                r_max,
                float(args.level),
            )
            rows.append(
                (
                    label,
                    alpha_r_deg,
                    alpha_i_deg,
                    gamma_deg,
                    math.degrees(float(beta)),
                    sector_label(float(beta), receiver, incident),
                    width / wavelength,
                )
            )

    npz_path = out_dir / "stationary_sector_widths.npz"
    np.savez_compressed(
        npz_path,
        case=np.asarray([row[0] for row in rows]),
        alpha_r_deg=np.asarray([row[1] for row in rows], dtype=float),
        alpha_i_deg=np.asarray([row[2] for row in rows], dtype=float),
        gamma_deg=np.asarray([row[3] for row in rows], dtype=float),
        beta_deg=np.asarray([row[4] for row in rows], dtype=float),
        sector=np.asarray([row[5] for row in rows]),
        first_crossing_over_lambda=np.asarray([row[6] for row in rows], dtype=float),
    )

    sector_colors = {"C": "tab:red", "M_I": "tab:orange", "M_R": "tab:blue", "G": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    for ax, label in zip(axes, ["equal_gamma_40", "unequal_gamma_40"]):
        subset = [row for row in rows if row[0] == label]
        for sector in ["C", "M_I", "M_R", "G"]:
            part = [row for row in subset if row[5] == sector]
            if not part:
                continue
            ax.scatter(
                [row[4] for row in part],
                [row[6] for row in part],
                s=18,
                color=sector_colors[sector],
                label=sector,
                alpha=0.85,
            )
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel(r"separation direction $\beta$ (degrees)")
        ax.set_ylabel(r"first 3 dB radius $R/\lambda$")
        ax.grid(True, alpha=0.3)
        ax.legend(title="sector", fontsize=8)
    fig.savefig(out_dir / "stationary_sector_widths.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "stationary_sector_widths.png")}


def gram_matrix(
    points: Array,
    k: float,
    receiver: ApertureArc,
    incident: ApertureArc | None,
    n_quad: int,
) -> CArray:
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    gram = np.eye(n_points, dtype=np.complex128)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            h = points[i] - points[j]
            radius = float(np.linalg.norm(h))
            beta = float(math.atan2(h[1], h[0]))
            val = product_factor(k, np.asarray([radius], dtype=float), beta, receiver, incident, n_quad)[0]
            gram[i, j] = val
            gram[j, i] = np.conjugate(val)
    return gram


def cumulative_coherence(gram: CArray) -> float:
    off_diag = np.abs(gram - np.eye(gram.shape[0], dtype=np.complex128))
    return float(np.max(np.sum(off_diag, axis=1)))


def run_gram_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alpha = math.radians(20.0)
    geometries = [
        ("single_stationary", ApertureArc(0.0, alpha, 1.0), None, 0.0),
        ("single_transverse", ApertureArc(0.0, alpha, 1.0), None, math.pi / 2.0),
        ("dual_orthogonal_stationary_for_receiver", ApertureArc(0.0, alpha, 1.0), ApertureArc(math.pi / 2.0, alpha, -1.0), 0.0),
        ("dual_coincident_stationary", ApertureArc(0.0, alpha, 1.0), ApertureArc(0.0, alpha, -1.0), 0.0),
    ]
    separations = np.linspace(0.5 * wavelength, 6.0 * wavelength, 18)
    rows: list[tuple[str, float, float, float, float, float]] = []
    for label, receiver, incident, beta in geometries:
        direction = np.asarray([math.cos(beta), math.sin(beta)], dtype=float)
        for sep in separations:
            two_points = np.vstack([np.zeros(2), sep * direction])
            gram = gram_matrix(two_points, k, receiver, incident, int(args.n_quad))
            mu = abs(gram[0, 1])
            cond = float(np.linalg.cond(gram))
            exact_two = (1.0 + mu) / max(1.0 - mu, 1e-14)
            rows.append((label, sep / wavelength, mu, cond, exact_two, cumulative_coherence(gram)))

    npz_path = out_dir / "gram_conditioning.npz"
    np.savez_compressed(
        npz_path,
        case=np.asarray([row[0] for row in rows]),
        separation_over_lambda=np.asarray([row[1] for row in rows], dtype=float),
        mu=np.asarray([row[2] for row in rows], dtype=float),
        condition_number=np.asarray([row[3] for row in rows], dtype=float),
        two_point_formula=np.asarray([row[4] for row in rows], dtype=float),
        nu1=np.asarray([row[5] for row in rows], dtype=float),
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for label, _, _, _ in geometries:
        subset = [row for row in rows if row[0] == label]
        xs = np.asarray([float(row[1]) for row in subset], dtype=float)
        ys = np.asarray([float(row[3]) for row in subset], dtype=float)
        ax.semilogy(xs, ys, "o-", label=label.replace("_", " "))
    ax.set_xlabel(r"separation $R/\lambda$")
    ax.set_ylabel(r"$\kappa_2(G_2)$")
    ax.set_title("Two-point Gram conditioning")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "gram_conditioning.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "gram_conditioning.png")}


def run_design_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alpha = math.radians(20.0)
    radius = 2.0 * wavelength
    relevant_betas = np.radians([0.0, 45.0, 90.0])
    centers = np.linspace(0.0, math.pi, 61)
    heatmap = np.empty((centers.size, centers.size), dtype=float)
    for i, theta_r in enumerate(centers):
        receiver = ApertureArc(theta_r, alpha, 1.0)
        for j, theta_i in enumerate(centers):
            incident = ApertureArc(theta_i, alpha, -1.0)
            vals = [
                abs(product_factor(k, np.asarray([radius]), float(beta), receiver, incident, int(args.n_quad))[0])
                for beta in relevant_betas
            ]
            heatmap[i, j] = max(vals)

    best = np.unravel_index(int(np.argmin(heatmap)), heatmap.shape)
    best_theta_r = float(centers[best[0]])
    best_theta_i = float(centers[best[1]])
    npz_path = out_dir / "aperture_design_scan.npz"
    np.savez_compressed(
        npz_path,
        centers=centers,
        heatmap=heatmap,
        alpha=alpha,
        radius=radius,
        relevant_betas=relevant_betas,
        best_theta_r=best_theta_r,
        best_theta_i=best_theta_i,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(
        heatmap,
        origin="lower",
        extent=[0.0, 180.0, 0.0, 180.0],
        cmap="viridis",
        aspect="auto",
    )
    ax.scatter([math.degrees(best_theta_i)], [math.degrees(best_theta_r)], c="red", marker="x", s=60)
    ax.set_xlabel(r"incident center $\theta_I$ (degrees)")
    ax.set_ylabel(r"receiver center $\theta_R$ (degrees)")
    ax.set_title(r"max coherence over $\mathcal{B}=\{0^\circ,45^\circ,90^\circ\}$")
    fig.colorbar(im, ax=ax, label="max product coherence")
    fig.savefig(out_dir / "aperture_design_scan.png", dpi=180)
    plt.close(fig)
    return {
        "npz": str(npz_path),
        "plot": str(out_dir / "aperture_design_scan.png"),
        "best_theta_r_deg": math.degrees(best_theta_r),
        "best_theta_i_deg": math.degrees(best_theta_i),
        "best_max_coherence": float(heatmap[best]),
    }


def run_noise_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alpha = math.radians(20.0)
    receiver = ApertureArc(0.0, alpha, 1.0)
    incident = ApertureArc(math.pi / 2.0, alpha, -1.0)
    cases = [
        ("well_conditioned", 4.0 * wavelength, math.pi / 2.0, receiver, None),
        ("nearly_coherent", 1.0 * wavelength, 0.0, receiver, None),
        ("dual_rescue", 1.0 * wavelength, 0.0, receiver, incident),
    ]
    eps_levels = [0.01, 0.05, 0.10]
    rng = np.random.default_rng(int(args.seed))
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for label, sep, beta, rec, inc in cases:
        direction = np.asarray([math.cos(beta), math.sin(beta)], dtype=float)
        points = np.vstack([np.zeros(2), sep * direction])
        gram = gram_matrix(points, k, rec, inc, int(args.n_quad))
        nu1 = cumulative_coherence(gram)
        inv_gram = np.linalg.inv(gram)
        bound_factor = 1.0 / max(1.0 - nu1, 1e-14)
        cond = float(np.linalg.cond(gram))
        for eps in eps_levels:
            for _ in range(int(args.noise_trials)):
                residual_normal = rng.normal(size=2) + 1j * rng.normal(size=2)
                residual_normal = residual_normal / np.linalg.norm(residual_normal) * eps
                coeff_error = inv_gram @ residual_normal
                err_norm = float(np.linalg.norm(coeff_error))
                rows.append((label, sep / wavelength, eps, nu1, cond, err_norm, bound_factor * eps))

    npz_path = out_dir / "noise_stability.npz"
    np.savez_compressed(
        npz_path,
        case=np.asarray([row[0] for row in rows]),
        separation_over_lambda=np.asarray([row[1] for row in rows], dtype=float),
        epsilon=np.asarray([row[2] for row in rows], dtype=float),
        nu1=np.asarray([row[3] for row in rows], dtype=float),
        condition_number=np.asarray([row[4] for row in rows], dtype=float),
        coefficient_error=np.asarray([row[5] for row in rows], dtype=float),
        bound=np.asarray([row[6] for row in rows], dtype=float),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for label, *_ in cases:
        subset = [row for row in rows if row[0] == label]
        xs = sorted({row[2] for row in subset})
        medians = []
        bounds = []
        for eps in xs:
            vals = [row[5] for row in subset if math.isclose(row[2], eps)]
            bvals = [row[6] for row in subset if math.isclose(row[2], eps)]
            medians.append(float(np.median(vals)))
            bounds.append(float(np.median(bvals)))
        ax.loglog(xs, medians, "o-", label=label.replace("_", " "))
        ax.loglog(xs, bounds, "--", color=ax.lines[-1].get_color(), alpha=0.65)
    ax.set_xlabel(r"$\|A^*e\|_2$")
    ax.set_ylabel(r"coefficient error norm")
    ax.set_title("Gram-controlled coefficient stability")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "noise_stability.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "noise_stability.png")}


def cap_center_factor(k: float, radii: Array, alpha: float) -> Array:
    argument = 0.5 * float(k) * np.asarray(radii, dtype=float) * (1.0 - math.cos(alpha))
    return np.abs(sin_over_x(argument))


def cap_transverse_factor(k: float, radii: Array, alpha: float) -> Array:
    argument = float(k) * np.asarray(radii, dtype=float) * alpha
    return np.abs(airy_j1_over_x(argument))


def run_cap3d_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    k = float(args.k)
    wavelength = PI2 / k
    alphas_deg = [10.0, 20.0, 30.0]
    rows: list[tuple[float, str, float, float]] = []
    profiles: dict[tuple[float, str], tuple[Array, Array]] = {}
    for alpha_deg in alphas_deg:
        alpha = math.radians(alpha_deg)
        center_prediction = SINGLE_TRANSVERSE_3DB * wavelength / (0.5 * (1.0 - math.cos(alpha)))
        transverse_prediction = 0.2573 * wavelength / alpha
        for regime, prediction, factor_fn in [
            ("cap_center", center_prediction, cap_center_factor),
            ("transverse", transverse_prediction, cap_transverse_factor),
        ]:
            r_max = max(6.0 * wavelength, 1.8 * prediction)
            radii = np.linspace(0.0, r_max, int(args.n_r))
            profile = factor_fn(k, radii, alpha)
            measured = first_crossing(radii, profile, float(args.level))
            rows.append((alpha_deg, regime, measured / wavelength, prediction / wavelength))
            profiles[(alpha_deg, regime)] = (radii / wavelength, profile)

    npz_path = out_dir / "spherical_cap_widths.npz"
    np.savez_compressed(
        npz_path,
        alpha_deg=np.asarray([row[0] for row in rows], dtype=float),
        regime=np.asarray([row[1] for row in rows]),
        measured_r_over_lambda=np.asarray([row[2] for row in rows], dtype=float),
        predicted_r_over_lambda=np.asarray([row[3] for row in rows], dtype=float),
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for regime, marker in [("cap_center", "o"), ("transverse", "s")]:
        subset = [row for row in rows if row[1] == regime]
        xs = np.asarray([row[0] for row in subset], dtype=float)
        ys = np.asarray([row[2] for row in subset], dtype=float)
        pred = np.asarray([row[3] for row in subset], dtype=float)
        ax.loglog(xs, ys, marker + "-", label=f"{regime} measured")
        ax.loglog(xs, pred, "--", color=ax.lines[-1].get_color(), alpha=0.65)
    ax.set_xlabel(r"cap half-angle $\alpha$ (degrees)")
    ax.set_ylabel(r"first 3 dB radius $R/\lambda$")
    ax.set_title("Spherical-cap aperture factors")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "spherical_cap_widths.png", dpi=180)
    plt.close(fig)
    return {"npz": str(npz_path), "plot": str(out_dir / "spherical_cap_widths.png")}


def selected_experiments(text: str) -> set[str]:
    names = {item.strip() for item in text.split(",") if item.strip()}
    if not names or "all" in names:
        return {"widths", "dual", "sectors", "gram", "noise", "design", "cap3d"}
    allowed = {"widths", "dual", "sectors", "gram", "noise", "design", "cap3d"}
    unknown = names - allowed
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(sorted(unknown))}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel-level limited-aperture coherence experiments.")
    parser.add_argument("--out-dir", default="outputs_limited_aperture_coherence")
    parser.add_argument("--experiments", default="all", help="Comma list: all,widths,dual,sectors,gram,noise,design,cap3d")
    parser.add_argument("--k", type=float, default=PI2)
    parser.add_argument("--level", type=float, default=THREE_DB_LEVEL)
    parser.add_argument("--n-quad", type=int, default=401)
    parser.add_argument("--n-r", type=int, default=600)
    parser.add_argument("--n-beta", type=int, default=91)
    parser.add_argument("--dual-alpha-deg", type=float, default=20.0)
    parser.add_argument("--noise-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260602)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = selected_experiments(str(args.experiments))
    outputs: dict[str, object] = {
        "model": "raw aperture coherence kernel and Gram matrix",
        "formula": "C_alpha(h)=(2 alpha)^-1 int exp(i k xhat.h) dtheta, mu_RI=|C_R C_I|",
        "k": float(args.k),
        "lambda": PI2 / float(args.k),
        "level": float(args.level),
        "n_quad": int(args.n_quad),
        "n_r": int(args.n_r),
        "n_beta": int(args.n_beta),
        "noise_trials": int(args.noise_trials),
        "seed": int(args.seed),
        "experiments": sorted(experiments),
    }

    if "widths" in experiments:
        outputs["widths"] = run_width_experiment(args, out_dir)
    if "dual" in experiments:
        outputs["dual"] = run_dual_experiment(args, out_dir)
    if "sectors" in experiments:
        outputs["sectors"] = run_sector_experiment(args, out_dir)
    if "gram" in experiments:
        outputs["gram"] = run_gram_experiment(args, out_dir)
    if "noise" in experiments:
        outputs["noise"] = run_noise_experiment(args, out_dir)
    if "design" in experiments:
        outputs["design"] = run_design_experiment(args, out_dir)
    if "cap3d" in experiments:
        outputs["cap3d"] = run_cap3d_experiment(args, out_dir)

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
