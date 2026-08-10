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


def polarimetric_block_from_directions(
    incident_dir: Array,
    obs_dir: Array,
    kind: TensorKind = "full",
) -> CArray:
    """Build the six-channel block for one direction configuration.

    The two columns of the incident polarization basis are perpendicular to
    ``incident_dir``.  Each tensor coefficient contributes three observed
    components for each polarization, hence six rows per configuration.
    """
    from common.phantom import tensor_basis
    from common.quadrature import orthonormal_basis_perp

    incident_dir = np.asarray(incident_dir, dtype=float)
    obs_dir = np.asarray(obs_dir, dtype=float)
    if incident_dir.shape != (3,) or obs_dir.shape != (3,):
        raise ValueError("incident_dir and obs_dir must have shape (3,)")

    e1, e2 = orthonormal_basis_perp(incident_dir)
    incident_polarizations = np.column_stack([e1, e2])
    observation_projection = np.eye(3) - np.outer(obs_dir, obs_dir)
    basis = tensor_basis(kind)
    block = np.zeros((6, len(basis)), dtype=np.complex128)
    for column, tensor in enumerate(basis):
        projected = observation_projection @ tensor @ incident_polarizations
        block[:, column] = projected.reshape(-1, order="F")
    return block


def build_polarimetric_matrix_from_directions(
    incident_dirs: Array,
    obs_dirs: Array,
    kind: TensorKind = "full",
) -> CArray:
    """Stack polarimetric blocks for explicitly supplied direction pairs."""
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if (
        incident_dirs.shape != obs_dirs.shape
        or incident_dirs.ndim != 2
        or incident_dirs.shape[1] != 3
    ):
        raise ValueError(
            "incident_dirs and obs_dirs must both have shape (n_configurations, 3)"
        )
    if incident_dirs.shape[0] == 0:
        raise ValueError("at least one direction configuration is required")
    return np.vstack([
        polarimetric_block_from_directions(d, xhat, kind)
        for d, xhat in zip(incident_dirs, obs_dirs)
    ])


def build_polarimetric_matrix(
    p: Array, kind: TensorKind = "full", J: int = 6
) -> CArray:
    """Build the polarimetric matrix ``M(p)`` from admissible geometries."""
    from common.quadrature import build_geometries_from_p

    geometries = build_geometries_from_p(p, J=J)
    incident_dirs = np.asarray([geometry[0] for geometry in geometries], dtype=float)
    obs_dirs = np.asarray([geometry[1] for geometry in geometries], dtype=float)
    return build_polarimetric_matrix_from_directions(incident_dirs, obs_dirs, kind)


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
