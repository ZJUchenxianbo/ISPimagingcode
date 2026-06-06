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
    from common.utils import complex_relative_noise

    p_nodes = np.asarray(p_nodes, dtype=float)
    true_coeffs = np.asarray(true_coeffs, dtype=np.complex128)
    recovered = np.zeros_like(true_coeffs)
    sigma_min = np.zeros(p_nodes.shape[0], dtype=float)
    cond_values = np.zeros(p_nodes.shape[0], dtype=float)
    for idx, (p, coeff) in enumerate(zip(p_nodes, true_coeffs)):
        M = build_polarimetric_matrix(p, kind=kind, J=J)
        singular = np.linalg.svd(M, compute_uv=False)
        sigma_min[idx] = float(np.min(singular))
        cond_values[idx] = float(np.max(singular) / max(np.min(singular), 1e-14))
        data = M @ coeff
        if noise_level > 0.0:
            data = data + complex_relative_noise(data, noise_level, rng)
        recovered[idx] = np.linalg.pinv(M) @ data
    return recovered, sigma_min, cond_values
