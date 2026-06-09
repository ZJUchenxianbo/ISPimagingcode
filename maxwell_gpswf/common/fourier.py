#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fourier-series utilities on the cube ``[-R, R]^3``.

These helpers provide a simple 09-003-style baseline for Maxwell-Born
experiments.  They use standard cube Fourier modes

    exp(i*pi*l.x/R),  l in Z^3,

and are intentionally separate from the ball GPSWF basis.  The baseline uses
Fourier coefficients of the tensor contrast after polarimetric recovery; in the
current analytical experiments those coefficients are generated directly from
the block phantom.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.phantom import Block
from common.utils import complex_relative_noise, vector_norm


def cube_fourier_indices(
    k_max: float,
    *,
    half_side: float = 1.0,
    bandwidth_factor: float = 2.0,
) -> np.ndarray:
    """Return integer lattice modes whose Fourier radius is within bandwidth.

    The cube is ``[-half_side, half_side]^3`` and the mode frequency is
    ``xi_l = pi*l/half_side``.  For comparison with the single-frequency GPSWF
    rows, the default Fourier radius is ``2*k_max`` because Maxwell-Born data at
    wave number ``k`` has Fourier bandwidth ``|xi| <= 2k``.
    """
    if k_max <= 0.0:
        raise ValueError("k_max must be positive")
    if half_side <= 0.0:
        raise ValueError("half_side must be positive")
    if bandwidth_factor <= 0.0:
        raise ValueError("bandwidth_factor must be positive")

    max_radius = float(bandwidth_factor) * float(k_max)
    max_index = int(math.floor(max_radius * float(half_side) / math.pi))
    values = np.arange(-max_index, max_index + 1, dtype=int)
    L1, L2, L3 = np.meshgrid(values, values, values, indexing="ij")
    indices = np.column_stack([L1.ravel(), L2.ravel(), L3.ravel()])
    frequencies = cube_fourier_frequencies(indices, half_side=half_side)
    keep = np.linalg.norm(frequencies, axis=1) <= max_radius + 1e-12
    kept = indices[keep]
    order = np.lexsort((kept[:, 2], kept[:, 1], kept[:, 0], np.sum(kept * kept, axis=1)))
    return kept[order]


def cube_fourier_frequencies(indices: np.ndarray, *, half_side: float = 1.0) -> np.ndarray:
    """Map integer cube modes to physical Fourier frequencies."""
    indices = np.asarray(indices, dtype=float)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape (n_modes, 3)")
    if half_side <= 0.0:
        raise ValueError("half_side must be positive")
    return (math.pi / float(half_side)) * indices


def block_fourier_transform_xi(xi: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Analytical transform ``int q(x) exp(-i xi.x) dx`` for block phantoms."""
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 2 or xi.shape[1] != 3:
        raise ValueError("xi must have shape (n_points, 3)")
    values = np.zeros(xi.shape[0], dtype=np.complex128)
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        phase = np.exp(-1j * (xi @ center))
        profile = np.prod(2.0 * half_width * np.sinc((xi * half_width) / math.pi), axis=1)
        values += complex(block.amplitude) * phase * profile
    return values


def cube_fourier_coefficients_from_blocks(
    blocks: list[Block],
    indices: np.ndarray,
    *,
    half_side: float = 1.0,
    component_value: complex = 1.0 + 0.0j,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return cube Fourier-series coefficients for one tensor component."""
    xi = cube_fourier_frequencies(indices, half_side=half_side)
    qhat = complex(component_value) * block_fourier_transform_xi(xi, blocks)
    volume = (2.0 * float(half_side)) ** 3
    coeffs = qhat / volume
    if noise_level > 0.0:
        if rng is None:
            raise ValueError("rng is required when noise_level > 0")
        coeffs = coeffs + complex_relative_noise(coeffs, float(noise_level), rng)
    return coeffs


def evaluate_cube_fourier_series(
    points: np.ndarray,
    indices: np.ndarray,
    coeffs: np.ndarray,
    *,
    half_side: float = 1.0,
    block_size: int = 2048,
) -> np.ndarray:
    """Evaluate ``sum_l coeff_l exp(i*pi*l.x/R)`` at points."""
    points = np.asarray(points, dtype=float)
    indices = np.asarray(indices, dtype=int)
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape (n_modes, 3)")
    if coeffs.shape != (indices.shape[0],):
        raise ValueError("coeffs must have shape (n_modes,)")
    xi = cube_fourier_frequencies(indices, half_side=half_side)
    values = np.empty(points.shape[0], dtype=np.complex128)
    for start in range(0, points.shape[0], int(block_size)):
        stop = min(start + int(block_size), points.shape[0])
        phase = np.exp(1j * (points[start:stop] @ xi.T))
        values[start:stop] = phase @ coeffs
    return values


def reconstruct_blocks_fourier_cube(
    blocks: list[Block],
    points: np.ndarray,
    k_max: float,
    *,
    half_side: float = 1.0,
    component_value: complex = 1.0 + 0.0j,
    bandwidth_factor: float = 2.0,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct one component of a block phantom using cube Fourier modes."""
    indices = cube_fourier_indices(
        k_max,
        half_side=half_side,
        bandwidth_factor=bandwidth_factor,
    )
    coeffs = cube_fourier_coefficients_from_blocks(
        blocks,
        indices,
        half_side=half_side,
        component_value=component_value,
        noise_level=noise_level,
        rng=rng,
    )
    image_values = evaluate_cube_fourier_series(points, indices, coeffs, half_side=half_side)
    xi = cube_fourier_frequencies(indices, half_side=half_side)
    meta = {
        "fourier_modes": int(indices.shape[0]),
        "fourier_index_linf": int(np.max(np.abs(indices))) if indices.size else 0,
        "fourier_radius_max": float(np.max(np.linalg.norm(xi, axis=1))) if xi.size else 0.0,
        "bandwidth_factor": float(bandwidth_factor),
        "coeff_norm": vector_norm(coeffs),
        "coeff_max_abs": float(np.max(np.abs(coeffs))) if coeffs.size else math.nan,
    }
    return image_values, meta
