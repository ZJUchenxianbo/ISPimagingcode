#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ball GPSWF (Generalized Prolate Spheroidal Wave Function) utilities.

Implements the tridiagonal eigenproblem from equation (4.3), radial
function evaluation, and alpha eigenvalue estimation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_jacobi, gammaln, spherical_jn

Array = NDArray[np.float64]

# Re-use table factory
from common.utils import make_table, vector_norm


# ---------------------------------------------------------------------------
# Tridiagonal system
# ---------------------------------------------------------------------------


def ball_gpswf_tridiagonal(C: float, ell: int, K: int) -> tuple[Array, Array]:
    """Return the tridiagonal matrix ``A_ell(C)`` from equation (4.3).

    The eigenproblem is ``A_ell(C) beta_{ell n} = chi_{ell n} beta_{ell n}``.
    """
    if C <= 0.0:
        raise ValueError("C must be positive")
    if ell < 0 or K < 0:
        raise ValueError("ell and K must be nonnegative")
    nu = ell + 0.5
    j = np.arange(K + 1, dtype=float)
    degree = ell + 2.0 * j
    gamma = degree * (degree + 3.0)
    b = nu * nu / ((2.0 * j + nu) * (2.0 * j + nu + 2.0))
    diag = gamma + 0.5 * C * C * (1.0 + b)
    if K == 0:
        return diag, np.empty(0, dtype=float)
    jj = np.arange(K, dtype=float)
    offdiag = (
        2.0
        * (jj + 1.0)
        * (jj + nu + 1.0)
        / ((2.0 * jj + nu + 2.0) * np.sqrt((2.0 * jj + nu + 1.0) * (2.0 * jj + nu + 3.0)))
    )
    offdiag *= 0.5 * C * C
    return diag, offdiag


def solve_ball_gpswf(
    C: float, ell: int, K: int, n_modes: int | None = None
) -> tuple[Array, Array]:
    """Solve for ``chi_{ell n}`` and ``beta_{ell n}`` in the ball GPSWF system."""
    diag, offdiag = ball_gpswf_tridiagonal(C, ell, K)
    if n_modes is None or n_modes >= K + 1:
        chi, beta = eigh_tridiagonal(diag, offdiag)
    else:
        chi, beta = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, int(n_modes) - 1))
    return chi, beta


def tridiagonal_residual(
    diag: Array, offdiag: Array, chi: float, beta: Array
) -> float:
    """Relative residual for a tridiagonal eigenpair."""
    Abeta = diag * beta
    if offdiag.size:
        Abeta[:-1] += offdiag * beta[1:]
        Abeta[1:] += offdiag * beta[:-1]
    return vector_norm(Abeta - chi * beta) / max(abs(float(chi)) * vector_norm(beta), 1e-300)


# ---------------------------------------------------------------------------
# Radial function evaluation
# ---------------------------------------------------------------------------


def jacobi_orthonormal_scale(j: int, nu: float) -> float:
    """Normalization for P_j^(0,nu) with weight (1+x)^nu on [-1,1]."""
    log_h = (
        (nu + 1.0) * math.log(2.0)
        - math.log(2.0 * j + nu + 1.0)
        + gammaln(j + 1.0)
        + gammaln(j + nu + 1.0)
        - gammaln(j + 1.0)
        - gammaln(j + nu + 1.0)
    )
    return math.exp(-0.5 * log_h)


def eval_radial_R(r: Array, ell: int, beta: Array) -> Array:
    """Evaluate the L²(B)-normalized radial factor ``R_{ell n}(r; C)``."""
    r = np.asarray(r, dtype=float)
    nu = ell + 0.5
    eta = 2.0 * r * r - 1.0
    values = np.zeros_like(r, dtype=float)
    for j, coeff in enumerate(beta):
        scale = jacobi_orthonormal_scale(j, nu)
        values += float(coeff) * scale * eval_jacobi(j, 0.0, nu, eta)
    radial_normalization = 2.0 ** (0.5 * float(ell) + 1.25)
    return radial_normalization * (r**ell) * values


# ---------------------------------------------------------------------------
# Alpha eigenvalue estimation
# ---------------------------------------------------------------------------


def compute_alpha_radial(
    C: float,
    ell: int,
    beta: Array,
    quad_order: int = 220,
    r_eval_count: int = 160,
) -> complex:
    """Estimate ``alpha_{ell n}(C)`` for the ``exp(+i C p.x)`` convention.

    Uses a weighted least-squares Rayleigh quotient over r nodes.  Intended
    for modal cutoff diagnostics, not as a high-precision GPSWF normalisation.
    """
    s_nodes, s_weights = np.polynomial.legendre.leggauss(quad_order)
    s = 0.5 * (s_nodes + 1.0)
    ws = 0.5 * s_weights
    R_s = eval_radial_R(s, ell, beta)

    r_nodes, r_weights = np.polynomial.legendre.leggauss(r_eval_count)
    r = 0.5 * (r_nodes + 1.0)
    wr = 0.5 * r_weights
    R_r = eval_radial_R(r, ell, beta)
    kernel = spherical_jn(ell, C * np.outer(r, s))
    F_R = 4.0 * math.pi * (1j**ell) * (kernel @ (ws * R_s * s * s))

    weight = wr * r * r
    mask = np.abs(R_r) > 1e-10 * max(float(np.max(np.abs(R_r))), 1.0)
    if not np.any(mask):
        mask = np.ones_like(R_r, dtype=bool)
    numerator = np.vdot(R_r[mask] * weight[mask], F_R[mask])
    denominator = np.vdot(R_r[mask] * weight[mask], R_r[mask])
    return complex(numerator / denominator)


def collect_alpha_pairs(
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    *,
    quad_order: int,
    r_eval_count: int,
) -> Any:
    """Compute alpha estimates for a rectangular set of ``(ell, n)`` pairs."""
    rows = []
    for ell in range(ell_max + 1):
        _, beta = solve_ball_gpswf(C, ell, K, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            alpha = compute_alpha_radial(
                C, ell, beta[:, n], quad_order=quad_order, r_eval_count=r_eval_count
            )
            rows.append(
                {
                    "ell": ell,
                    "n": n,
                    "alpha_real": float(np.real(alpha)),
                    "alpha_imag": float(np.imag(alpha)),
                    "alpha_abs": float(abs(alpha)),
                }
            )
    return make_table(rows)


# ---------------------------------------------------------------------------
# Alpha cache
# ---------------------------------------------------------------------------


def _alpha_float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _alpha_cache_path(
    cache_dir: Path,
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    quad_order: int,
    r_eval_count: int,
) -> Path:
    return cache_dir / (
        f"alpha_C{_alpha_float_token(C)}_K{int(K)}_ell{int(ell_max)}_"
        f"modes{int(n_modes_per_ell)}_q{int(quad_order)}_r{int(r_eval_count)}.npz"
    )


def _alpha_cache_candidates(
    cache_dir: Path, C: float, K: int, quad_order: int, r_eval_count: int
) -> list[Path]:
    prefix = _alpha_float_token(C)
    files = list(
        cache_dir.glob(
            f"alpha_C{prefix}_K{int(K)}_ell*_modes*_q{int(quad_order)}_r{int(r_eval_count)}.npz"
        )
    )
    return sorted(files, key=lambda p: p.name)


def _save_alpha_cache(
    df: Any,
    path: Path,
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    quad_order: int,
    r_eval_count: int,
) -> None:
    alpha_real = df["alpha_real"].to_numpy(dtype=float)
    alpha_imag = df["alpha_imag"].to_numpy(dtype=float)
    np.savez_compressed(
        path,
        C=np.asarray([float(C)]),
        K=np.asarray([int(K)]),
        ell_max=np.asarray([int(ell_max)]),
        n_modes_per_ell=np.asarray([int(n_modes_per_ell)]),
        quad_order=np.asarray([int(quad_order)]),
        r_eval_count=np.asarray([int(r_eval_count)]),
        ell=df["ell"].to_numpy(dtype=int),
        n=df["n"].to_numpy(dtype=int),
        alpha_real=alpha_real,
        alpha_imag=alpha_imag,
        alpha_abs=np.abs(alpha_real + 1j * alpha_imag),
    )


def _load_alpha_cache_subset(
    path: Path,
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    quad_order: int,
    r_eval_count: int,
) -> list[dict[str, object]] | None:
    try:
        data = np.load(path)
    except OSError:
        return None
    with data:
        if abs(float(data["C"][0]) - float(C)) > 1e-12:
            return None
        if int(data["K"][0]) != int(K):
            return None
        if int(data["quad_order"][0]) != int(quad_order):
            return None
        if int(data["r_eval_count"][0]) != int(r_eval_count):
            return None
        if int(data["ell_max"][0]) < int(ell_max):
            return None
        if int(data["n_modes_per_ell"][0]) < int(n_modes_per_ell):
            return None

        ell = data["ell"].astype(int)
        n = data["n"].astype(int)
        alpha_real = data["alpha_real"].astype(float)
        alpha_imag = data["alpha_imag"].astype(float)
        mask = (ell <= int(ell_max)) & (n < int(n_modes_per_ell))
        rows = []
        for ell_j, n_j, real_j, imag_j in zip(
            ell[mask], n[mask], alpha_real[mask], alpha_imag[mask]
        ):
            alpha = complex(float(real_j), float(imag_j))
            rows.append(
                {
                    "ell": int(ell_j),
                    "n": int(n_j),
                    "alpha_real": float(real_j),
                    "alpha_imag": float(imag_j),
                    "alpha_abs": float(abs(alpha)),
                }
            )
        return rows


def collect_alpha_pairs_cached(
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    *,
    quad_order: int,
    r_eval_count: int,
    cache_dir: Path,
) -> Any:
    """Load alpha estimates from cache, or compute and cache them.

    A cache with larger ``ell_max`` and ``n_modes_per_ell`` can satisfy a
    smaller request when the remaining numerical parameters match.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for candidate in _alpha_cache_candidates(cache_dir, C, K, quad_order, r_eval_count):
        rows = _load_alpha_cache_subset(
            candidate, C, K, ell_max, n_modes_per_ell, quad_order, r_eval_count
        )
        if rows is not None:
            return make_table(rows)

    df = collect_alpha_pairs(
        C, K, ell_max, n_modes_per_ell, quad_order=quad_order, r_eval_count=r_eval_count
    )
    _save_alpha_cache(
        df,
        _alpha_cache_path(cache_dir, C, K, ell_max, n_modes_per_ell, quad_order, r_eval_count),
        C, K, ell_max, n_modes_per_ell, quad_order, r_eval_count,
    )
    return df


# ---------------------------------------------------------------------------
# GPSWF modal matrix and quadrature projection (shared by reconstruction scripts)
# ---------------------------------------------------------------------------


def spherical_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (x,y,z) to (r, theta, phi)."""
    points = np.asarray(points, dtype=float)
    r = np.linalg.norm(points, axis=1)
    theta = np.zeros_like(r)
    nonzero = r > 1e-14
    theta[nonzero] = np.arccos(np.clip(points[nonzero, 2] / r[nonzero], -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    return r, theta, phi


def modal_matrix(
    points: np.ndarray,
    modes: list[Any],
    *,
    fourier_side: bool = True,
) -> np.ndarray:
    """Build the GPSWF evaluation matrix at given points.

    Parameters
    ----------
    fourier_side : bool
        If True, include ``conj(alpha)`` factor for the Fourier-domain basis
        (unified ``exp(-i C p·x)`` convention).  Set to False only for
        physical-domain (image) reconstruction.
    """
    from scipy.special import sph_harm_y

    r, theta, phi = spherical_coordinates(points)
    values = np.zeros((points.shape[0], len(modes)), dtype=np.complex128)
    for j, mode in enumerate(modes):
        radial = eval_radial_R(r, mode.ell, mode.beta)
        # scipy >= 1.17: sph_harm_y(n, m, theta, phi) with n=degree first
        # scipy <  1.17: sph_harm_y(m, n, theta, phi) with m=order  first
        # We use the >=1.17 convention: degree (mode.ell) comes first
        angular = sph_harm_y(mode.ell, mode.m, theta, phi)
        column = radial * angular
        if fourier_side:
            column = np.conj(mode.alpha) * column
        values[:, j] = column
    return values


def quadrature_modal_coefficients(
    component_data: np.ndarray,
    basis_matrix: np.ndarray,
    target_weights: np.ndarray,
    modes: list[Any],
    retained: np.ndarray,
) -> np.ndarray:
    """Compute GPSWF modal coefficients via regularised least-squares.

    Solves  (A^H W A + λ I) c = A^H W d  where A has columns
    ``conj(α_j) ψ_j(p)`` and W = diag(target_weights).

    The diagonal projection ``c_j = (A^H W d)_j / conj(α_j)`` is used as
    a fast approximation only when the quadrature is exact enough to make
    A^H W A nearly diagonal.  Otherwise the regularised normal equations
    are solved.
    """
    alpha = np.asarray([mode.alpha for mode in modes], dtype=np.complex128)
    A = basis_matrix[:, retained]
    n_retained = int(np.sum(retained))

    # Build weighted normal-equation matrix and right-hand side
    WA = (target_weights[:, None]) * A  # diag(W) @ A
    AWA = np.conj(A).T @ WA             # A^H W A
    AWd = np.conj(A).T @ (target_weights * component_data)  # A^H W d

    # Check whether the diagonal approximation is sufficient
    diag_AWA = np.diag(np.diag(AWA))
    offdiag = AWA - diag_AWA
    rel_offdiag = np.linalg.norm(offdiag) / max(np.linalg.norm(diag_AWA), 1e-14)

    if rel_offdiag < 0.01:
        # Quadrature is near-exact — diagonal projection is fine.
        # (A^H W A)_{jj} ≈ |α_j|², so c_j ≈ (A^H W d)_j / |α_j|².
        retained_coeffs = AWd / (np.abs(alpha[retained]) ** 2)
    else:
        # Regularised least-squares
        reg = 1e-10 * np.abs(np.trace(AWA)) / max(n_retained, 1)
        retained_coeffs = np.linalg.solve(
            AWA + reg * np.eye(n_retained, dtype=np.complex128), AWd
        )

    coeffs = np.zeros(len(modes), dtype=np.complex128)
    coeffs[retained] = retained_coeffs
    return coeffs
