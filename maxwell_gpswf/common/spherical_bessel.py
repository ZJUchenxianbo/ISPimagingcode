#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spherical-harmonic Bessel basis utilities on the unit ball.

The basis functions are

    phi_{ell,n,m}(x) = N_{ell,n} j_ell(rho_{ell,n} |x|)
                       Y_ell^m(theta, phi),

where ``rho_{ell,n}`` is the n-th positive zero of ``j_ell``.  They form the
standard Dirichlet Laplacian eigenbasis on the unit ball.  In this project they
serve as a ball-basis baseline distinct from GPSWF: coefficients can be
recovered from the same finite Fourier data by a least-squares data equation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

from common.phantom import Block
from common.gpswf import spherical_coordinates
from common.utils import complex_relative_noise, vector_norm, weighted_lstsq


@dataclass(frozen=True)
class BesselMode:
    """Unit-ball spherical-harmonic Bessel mode identifier."""

    ell: int
    n: int
    m: int
    rho: float
    normalization: float


def spherical_bessel_roots(ell: int, cutoff: float, *, samples_per_unit: int = 160) -> np.ndarray:
    """Return positive zeros of ``j_ell`` not exceeding ``cutoff``."""
    if ell < 0:
        raise ValueError("ell must be nonnegative")
    if cutoff <= 0.0:
        return np.empty(0, dtype=float)

    scan_start = max(1e-8, 0.5 * float(ell))
    if scan_start >= float(cutoff):
        return np.empty(0, dtype=float)

    sample_count = max(int(math.ceil((float(cutoff) - scan_start) * int(samples_per_unit))), 200)
    grid = np.linspace(scan_start, float(cutoff), sample_count + 1)
    values = spherical_jn(int(ell), grid)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not (np.isfinite(f_left) and np.isfinite(f_right)):
            continue
        if abs(float(f_left)) < 1e-13:
            root = float(left)
        elif float(f_left) * float(f_right) < 0.0:
            root = float(brentq(lambda x: spherical_jn(int(ell), x), float(left), float(right)))
        elif abs(float(f_right)) < 1e-13:
            root = float(right)
        else:
            continue
        if root > 1e-8 and root <= float(cutoff) + 1e-10:
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
    return np.asarray(roots, dtype=float)


def ball_bessel_modes(
    k_max: float,
    *,
    bandwidth_factor: float = 2.0,
    ell_max: int | None = None,
    max_modes: int | None = None,
) -> list[BesselMode]:
    """Build unit-ball Bessel modes with ``rho_{ell,n} <= bandwidth_factor*k_max``."""
    if k_max <= 0.0:
        raise ValueError("k_max must be positive")
    if bandwidth_factor <= 0.0:
        raise ValueError("bandwidth_factor must be positive")

    rho_cutoff = float(bandwidth_factor) * float(k_max)
    if ell_max is None:
        # The first positive zero of j_ell is larger than ell, so larger
        # angular degrees cannot contribute below this radial cutoff.
        ell_max = int(math.floor(rho_cutoff))

    modes: list[BesselMode] = []
    for ell in range(int(ell_max) + 1):
        roots = spherical_bessel_roots(ell, rho_cutoff)
        for root_index, rho in enumerate(roots, start=1):
            next_value = spherical_jn(ell + 1, rho)
            normalization = math.sqrt(2.0) / max(abs(float(next_value)), 1e-300)
            for m in range(-ell, ell + 1):
                modes.append(
                    BesselMode(
                        ell=ell,
                        n=root_index,
                        m=m,
                        rho=float(rho),
                        normalization=float(normalization),
                    )
                )
    modes.sort(key=lambda mode: (mode.rho, mode.ell, mode.n, mode.m))
    if max_modes is not None and int(max_modes) > 0 and len(modes) > int(max_modes):
        return modes[: int(max_modes)]
    return modes


def ball_bessel_matrix(points: np.ndarray, modes: list[BesselMode]) -> np.ndarray:
    """Evaluate ball Bessel modes at ``points``."""
    from scipy.special import sph_harm_y

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    r, theta, phi = spherical_coordinates(points)
    inside = r <= 1.0 + 1e-12
    values = np.zeros((points.shape[0], len(modes)), dtype=np.complex128)
    for col, mode in enumerate(modes):
        radial = spherical_jn(mode.ell, mode.rho * r)
        angular = sph_harm_y(mode.ell, mode.m, theta, phi)
        column = mode.normalization * radial * angular
        column[~inside] = 0.0
        values[:, col] = column
    return values


def block_values_at_points(
    points: np.ndarray,
    blocks: list[Block],
    *,
    component_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Evaluate one tensor component of a block phantom at physical points."""
    points = np.asarray(points, dtype=float)
    values = np.zeros(points.shape[0], dtype=np.complex128)
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        inside = np.all(np.abs(points - center[None, :]) <= half_width[None, :], axis=1)
        values[inside] += complex(block.amplitude) * complex(component_value)
    return values


def ball_bessel_coefficients_from_blocks(
    blocks: list[Block],
    modes: list[BesselMode],
    projection_nodes: np.ndarray,
    projection_weights: np.ndarray,
    *,
    component_value: complex = 1.0 + 0.0j,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Project a block phantom onto the ball Bessel basis by volume quadrature."""
    basis = ball_bessel_matrix(projection_nodes, modes)
    values = block_values_at_points(
        projection_nodes,
        blocks,
        component_value=component_value,
    )
    coeffs = np.conj(basis).T @ (np.asarray(projection_weights, dtype=float) * values)
    if noise_level > 0.0:
        if rng is None:
            raise ValueError("rng is required when noise_level > 0")
        coeffs = coeffs + complex_relative_noise(coeffs, float(noise_level), rng)
    return coeffs


def ball_bessel_data_matrix(
    p_nodes: np.ndarray,
    modes: list[BesselMode],
    C: float,
    *,
    quadrature_nodes: np.ndarray,
    quadrature_weights: np.ndarray,
    block_size: int = 256,
) -> np.ndarray:
    """Return ``int_ball phi_j(x) exp(-i C p.x) dx`` for Bessel modes."""
    p_nodes = np.asarray(p_nodes, dtype=float)
    quadrature_nodes = np.asarray(quadrature_nodes, dtype=float)
    quadrature_weights = np.asarray(quadrature_weights, dtype=float)
    if p_nodes.ndim != 2 or p_nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if quadrature_nodes.ndim != 2 or quadrature_nodes.shape[1] != 3:
        raise ValueError("quadrature_nodes must have shape (n_quad, 3)")
    if quadrature_weights.shape != (quadrature_nodes.shape[0],):
        raise ValueError("quadrature_weights must have shape (n_quad,)")

    basis = ball_bessel_matrix(quadrature_nodes, modes)
    weighted_basis = quadrature_weights[:, None] * basis
    matrix = np.empty((p_nodes.shape[0], len(modes)), dtype=np.complex128)
    for start in range(0, p_nodes.shape[0], int(block_size)):
        stop = min(start + int(block_size), p_nodes.shape[0])
        phase = np.exp(-1j * float(C) * (p_nodes[start:stop] @ quadrature_nodes.T))
        matrix[start:stop] = phase @ weighted_basis
    return matrix


def reconstruct_ball_bessel_from_data(
    component_data: np.ndarray,
    p_nodes: np.ndarray,
    data_weights: np.ndarray,
    points: np.ndarray,
    k_max: float,
    C: float,
    *,
    quadrature_nodes: np.ndarray,
    quadrature_weights: np.ndarray,
    bandwidth_factor: float = 2.0,
    max_modes: int | None = None,
    rcond: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct one component from recovered Fourier data using Bessel modes."""
    candidate_modes = len(ball_bessel_modes(k_max, bandwidth_factor=bandwidth_factor))
    modes = ball_bessel_modes(k_max, bandwidth_factor=bandwidth_factor, max_modes=max_modes)
    data_matrix = ball_bessel_data_matrix(
        p_nodes,
        modes,
        C,
        quadrature_nodes=quadrature_nodes,
        quadrature_weights=quadrature_weights,
    )
    coeffs, ls_meta = weighted_lstsq(
        data_matrix,
        np.asarray(component_data, dtype=np.complex128),
        weights=np.asarray(data_weights, dtype=float),
        rcond=rcond,
    )
    image_matrix = ball_bessel_matrix(points, modes)
    image_values = image_matrix @ coeffs
    rho_values = np.asarray([mode.rho for mode in modes], dtype=float)
    meta = {
        "bessel_modes": int(len(modes)),
        "bessel_candidate_modes": int(candidate_modes),
        "basis_mode_cap": int(max_modes) if max_modes is not None else int(candidate_modes),
        "bessel_ell_max": int(max((mode.ell for mode in modes), default=-1)),
        "bessel_n_max": int(max((mode.n for mode in modes), default=0)),
        "bessel_rho_max": float(np.max(rho_values)) if rho_values.size else math.nan,
        "bandwidth_factor": float(bandwidth_factor),
        "coeff_norm": vector_norm(coeffs),
        "coeff_max_abs": float(np.max(np.abs(coeffs))) if coeffs.size else math.nan,
        "data_norm": vector_norm(np.asarray(component_data, dtype=np.complex128)),
        "data_max_abs": float(np.max(np.abs(component_data))) if np.asarray(component_data).size else math.nan,
        **ls_meta,
    }
    return image_values, meta


def reconstruct_blocks_ball_bessel(
    blocks: list[Block],
    points: np.ndarray,
    k_max: float,
    *,
    projection_nodes: np.ndarray,
    projection_weights: np.ndarray,
    component_value: complex = 1.0 + 0.0j,
    bandwidth_factor: float = 2.0,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct one component with truncated ball Bessel modes."""
    modes = ball_bessel_modes(k_max, bandwidth_factor=bandwidth_factor)
    coeffs = ball_bessel_coefficients_from_blocks(
        blocks,
        modes,
        projection_nodes,
        projection_weights,
        component_value=component_value,
        noise_level=noise_level,
        rng=rng,
    )
    image_matrix = ball_bessel_matrix(points, modes)
    image_values = image_matrix @ coeffs
    rho_values = np.asarray([mode.rho for mode in modes], dtype=float)
    meta = {
        "bessel_modes": int(len(modes)),
        "bessel_ell_max": int(max((mode.ell for mode in modes), default=-1)),
        "bessel_n_max": int(max((mode.n for mode in modes), default=0)),
        "bessel_rho_max": float(np.max(rho_values)) if rho_values.size else math.nan,
        "bandwidth_factor": float(bandwidth_factor),
        "coeff_norm": vector_norm(coeffs),
        "coeff_max_abs": float(np.max(np.abs(coeffs))) if coeffs.size else math.nan,
    }
    return image_values, meta
