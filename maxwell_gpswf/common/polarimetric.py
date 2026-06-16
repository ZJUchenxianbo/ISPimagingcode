#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polarimetric matrix M(p) and tensor Fourier coefficient recovery."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
TensorKind = Literal["full", "reciprocal", "isotropic"]


def build_polarimetric_matrix(
    p: Array, kind: TensorKind = "full", J: int = 6
) -> CArray:
    """Build the polarimetric matrix ``M(p)`` from admissible geometries."""
    from common.phantom import tensor_basis
    from common.quadrature import build_geometries_from_p

    basis = tensor_basis(kind)
    geometries = build_geometries_from_p(p, J=J)
    M = np.zeros((6 * J, len(basis)), dtype=np.complex128)
    I = np.eye(3)
    for r, T in enumerate(basis):
        column_blocks = []
        for _, xhat, E in geometries:
            P_xhat = I - np.outer(xhat, xhat)
            B = P_xhat @ T @ E
            column_blocks.append(B.reshape(-1, order="F"))
        M[:, r] = np.concatenate(column_blocks)
    return M


def recover_polarimetric_coefficients(
    p_nodes: Array,
    true_coeffs: CArray,
    kind: TensorKind,
    noise_level: float,
    rng: np.random.Generator,
    *,
    J: int = 6,
) -> tuple[CArray, Array, Array]:
    """Recover tensor Fourier coefficients from (noisy) polarimetric data.

    Noise is added to ``g(p) = M(p) c(p)`` before pseudo-inversion.
    """
    data, sigma_min, cond_values = polarimetric_farfield_data(
        p_nodes,
        true_coeffs,
        kind,
        noise_level,
        rng,
        J=J,
    )
    recovered = recover_polarimetric_coefficients_from_data(
        p_nodes,
        data,
        kind,
        J=J,
    )
    return recovered, sigma_min, cond_values


def polarimetric_farfield_data(
    p_nodes: Array,
    true_coeffs: CArray,
    kind: TensorKind,
    noise_level: float,
    rng: np.random.Generator,
    *,
    J: int = 6,
) -> tuple[CArray, Array, Array]:
    """Generate noisy Maxwell far-field channel data ``g(p)=M(p)c(p)``.

    The returned data has shape ``(n_nodes, 6*J)``.  For each Fourier node,
    the ``6*J`` channels are the two incident polarizations and three observed
    vector components for each of the ``J`` admissible geometries.
    """
    from common.utils import complex_relative_noise

    p_nodes = np.asarray(p_nodes, dtype=float)
    true_coeffs = np.asarray(true_coeffs, dtype=np.complex128)
    if true_coeffs.ndim != 2 or true_coeffs.shape[0] != p_nodes.shape[0]:
        raise ValueError("true_coeffs must have shape (n_nodes, n_coeffs)")

    data = np.zeros((p_nodes.shape[0], 6 * int(J)), dtype=np.complex128)
    sigma_min = np.zeros(p_nodes.shape[0], dtype=float)
    cond_values = np.zeros(p_nodes.shape[0], dtype=float)
    for idx, (p, coeff) in enumerate(zip(p_nodes, true_coeffs)):
        M = build_polarimetric_matrix(p, kind=kind, J=J)
        singular = np.linalg.svd(M, compute_uv=False)
        sigma_min[idx] = float(np.min(singular))
        cond_values[idx] = float(np.max(singular) / max(np.min(singular), 1e-14))
        datum = M @ coeff
        if noise_level > 0.0:
            datum = datum + complex_relative_noise(datum, noise_level, rng)
        data[idx] = datum
    return data, sigma_min, cond_values


def recover_polarimetric_coefficients_from_data(
    p_nodes: Array,
    farfield_data: CArray,
    kind: TensorKind,
    *,
    J: int = 6,
) -> CArray:
    """Recover tensor Fourier coefficients from stored far-field data."""
    p_nodes = np.asarray(p_nodes, dtype=float)
    farfield_data = np.asarray(farfield_data, dtype=np.complex128)
    if farfield_data.shape != (p_nodes.shape[0], 6 * int(J)):
        raise ValueError("farfield_data must have shape (n_nodes, 6*J)")

    if p_nodes.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    n_coeffs = build_polarimetric_matrix(p_nodes[0], kind=kind, J=J).shape[1]
    recovered = np.zeros((p_nodes.shape[0], n_coeffs), dtype=np.complex128)
    for idx, (p, datum) in enumerate(zip(p_nodes, farfield_data)):
        M = build_polarimetric_matrix(p, kind=kind, J=J)
        recovered[idx] = np.linalg.pinv(M) @ datum
    return recovered
