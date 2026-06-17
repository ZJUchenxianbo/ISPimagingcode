#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIM-GPSWF helpers for scalar contrast updates.

The first nonlinear prototype uses

    Q(x) = q(x) T0

with a fixed tensor ``T0`` and represents the perturbation ``delta q`` in the
retained GPSWF space.  This keeps the nonlinear correction aligned with the
existing low-rank GPSWF reconstruction pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from common import orthonormal_basis_perp, tensor_coefficients_from_matrix
from common.phantom import Block
from common.utils import vector_norm
from forward.vie import (
    assemble_vie_matrix,
    incident_plane_wave,
    maxwell_far_field,
    vie_to_fourier_convention,
)


@dataclass(frozen=True)
class ScalarVIEData:
    """Scalarized Maxwell VIE data for one set of direction pairs."""

    data: np.ndarray
    total_fields: np.ndarray | None
    gmres_info_max: int
    relative_residual_max: float
    matrix_residual: float


@dataclass(frozen=True)
class RawVIEData:
    """Normalized raw Maxwell far-field channel data."""

    farfield_data: np.ndarray
    total_fields: np.ndarray | None
    matrix_residual: float


@dataclass(frozen=True)
class BIMLinearization:
    """Linearized BIM response in retained GPSWF coordinates."""

    matrix: np.ndarray
    matrix_norm: float
    condition: float


def evaluate_blocks_on_nodes(nodes: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Evaluate the scalar block amplitude ``q`` on volume nodes."""
    nodes = np.asarray(nodes, dtype=float)
    values = np.zeros(nodes.shape[0], dtype=np.complex128)
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        inside = np.all(np.abs(nodes - center[None, :]) <= half_width[None, :], axis=1)
        values[inside] += complex(block.amplitude)
    return values


def build_scalar_contrast(q_values: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    """Build voxel contrast ``Q_i = q_i T0``."""
    q = np.asarray(q_values, dtype=np.complex128)
    tensor = np.asarray(tensor, dtype=np.complex128)
    if tensor.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")
    return q[:, None, None] * tensor[None, :, :]


def _farfield_prefactor(k: float) -> float:
    return float(k) ** 2 / (4.0 * math.pi)


def compute_scalar_vie_data(
    *,
    p_nodes: np.ndarray,
    incident_dirs: np.ndarray,
    obs_dirs: np.ndarray,
    volume_nodes: np.ndarray,
    volume_weights: np.ndarray,
    q_values: np.ndarray,
    tensor: np.ndarray,
    k: float,
    h: float,
    return_fields: bool = False,
    conjugate_to_unified: bool = True,
) -> ScalarVIEData:
    """Compute scalarized Full-VIE data.

    With direction pairs satisfying ``p=(d-xhat)/2``, the VIE far field carries
    the ``exp(+i C p.x)`` phase.  The legacy path sets
    ``conjugate_to_unified=True`` and returns its complex conjugate.  When the
    caller instead constructs direction pairs for ``-p``, set this flag to
    ``False``; the raw VIE data is then already in the project
    ``exp(-i C p.x)`` convention for the positive target node ``p``.
    """
    del p_nodes  # Direction pairs already encode the Fourier nodes.
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    volume_nodes = np.asarray(volume_nodes, dtype=float)
    volume_weights = np.asarray(volume_weights, dtype=float)
    tensor = np.asarray(tensor, dtype=np.complex128)
    q_values = np.asarray(q_values, dtype=np.complex128)

    Q = build_scalar_contrast(q_values, tensor)
    A = assemble_vie_matrix(volume_nodes, volume_weights, Q, float(k), h=float(h))
    lu = lu_factor(A)

    n_pairs = incident_dirs.shape[0]
    n_voxels = volume_nodes.shape[0]
    scalar_data_vie = np.zeros(n_pairs, dtype=np.complex128)
    total_fields = (
        np.zeros((n_pairs, 2, n_voxels, 3), dtype=np.complex128)
        if return_fields else None
    )

    scale = 4.0 * math.pi / (float(k) ** 2)
    comp_scale = _tensor_component_scale(tensor)
    matrix_residual = 0.0

    for idx, (d, xhat) in enumerate(zip(incident_dirs, obs_dirs)):
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        projector = np.eye(3) - np.outer(xhat, xhat)
        numerator = 0.0 + 0.0j
        denominator = 0.0 + 0.0j
        for pol_idx in range(E_basis.shape[1]):
            e = E_basis[:, pol_idx].astype(np.complex128)
            rhs = incident_plane_wave(volume_nodes, float(k), d, e).reshape(-1)
            solution = lu_solve(lu, rhs)
            residual = vector_norm(A @ solution - rhs) / max(vector_norm(rhs), 1e-14)
            matrix_residual = max(matrix_residual, residual)
            field = solution.reshape((n_voxels, 3))
            if total_fields is not None:
                total_fields[idx, pol_idx] = field

            farfield = maxwell_far_field(
                volume_nodes,
                volume_weights,
                Q,
                field,
                float(k),
                xhat[None, :],
            )[0]
            model = projector @ tensor @ e
            numerator += np.vdot(model, scale * farfield)
            denominator += np.vdot(model, model)

        if abs(denominator) > 1e-14:
            scalar_data_vie[idx] = numerator / denominator * comp_scale

    data = (
        vie_to_fourier_convention(scalar_data_vie)
        if conjugate_to_unified
        else scalar_data_vie
    )
    return ScalarVIEData(
        data=data,
        total_fields=total_fields,
        gmres_info_max=0,
        relative_residual_max=0.0,
        matrix_residual=float(matrix_residual),
    )


def compute_raw_vie_farfield_data(
    *,
    incident_dirs: np.ndarray,
    obs_dirs: np.ndarray,
    volume_nodes: np.ndarray,
    volume_weights: np.ndarray,
    q_values: np.ndarray,
    tensor: np.ndarray,
    k: float,
    h: float,
    return_fields: bool = False,
    conjugate_to_unified: bool = False,
) -> RawVIEData:
    """Compute normalized raw Full-VIE far-field channels.

    The returned ``farfield_data`` is divided by ``k^2/(4π)``.  If the caller
    constructs physical direction pairs for ``-p``, keep
    ``conjugate_to_unified=False`` so the data follows the project
    ``exp(-i C p.x)`` convention.
    """
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    volume_nodes = np.asarray(volume_nodes, dtype=float)
    volume_weights = np.asarray(volume_weights, dtype=float)
    tensor = np.asarray(tensor, dtype=np.complex128)
    q_values = np.asarray(q_values, dtype=np.complex128)

    Q = build_scalar_contrast(q_values, tensor)
    A = assemble_vie_matrix(volume_nodes, volume_weights, Q, float(k), h=float(h))
    lu = lu_factor(A)

    n_pairs = incident_dirs.shape[0]
    n_voxels = volume_nodes.shape[0]
    farfield_data = np.zeros((n_pairs, 6), dtype=np.complex128)
    total_fields = (
        np.zeros((n_pairs, 2, n_voxels, 3), dtype=np.complex128)
        if return_fields else None
    )
    prefactor = _farfield_prefactor(k)
    matrix_residual = 0.0

    for idx, (d, xhat) in enumerate(zip(incident_dirs, obs_dirs)):
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        for pol_idx in range(E_basis.shape[1]):
            e = E_basis[:, pol_idx].astype(np.complex128)
            rhs = incident_plane_wave(volume_nodes, float(k), d, e).reshape(-1)
            solution = lu_solve(lu, rhs)
            residual = vector_norm(A @ solution - rhs) / max(vector_norm(rhs), 1e-14)
            matrix_residual = max(matrix_residual, residual)
            field = solution.reshape((n_voxels, 3))
            if total_fields is not None:
                total_fields[idx, pol_idx] = field

            farfield = maxwell_far_field(
                volume_nodes,
                volume_weights,
                Q,
                field,
                float(k),
                xhat[None, :],
            )[0] / prefactor
            if conjugate_to_unified:
                farfield = np.conj(farfield)
            farfield_data[idx, pol_idx * 3:(pol_idx + 1) * 3] = farfield

    return RawVIEData(
        farfield_data=farfield_data,
        total_fields=total_fields,
        matrix_residual=float(matrix_residual),
    )


def compute_bim_gpswf_linearization(
    *,
    incident_dirs: np.ndarray,
    obs_dirs: np.ndarray,
    volume_nodes: np.ndarray,
    volume_weights: np.ndarray,
    total_fields: np.ndarray,
    retained_mode_values: np.ndarray,
    tensor: np.ndarray,
    k: float,
    conjugate_to_unified: bool = True,
) -> BIMLinearization:
    """Build the retained GPSWF BIM response matrix.

    Column ``j`` is the scalarized far-field response generated by
    ``delta q = psi_j`` while the current total field is fixed.
    """
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    volume_nodes = np.asarray(volume_nodes, dtype=float)
    volume_weights = np.asarray(volume_weights, dtype=float)
    total_fields = np.asarray(total_fields, dtype=np.complex128)
    phi = np.asarray(retained_mode_values, dtype=np.complex128)
    tensor = np.asarray(tensor, dtype=np.complex128)
    if phi.ndim != 2:
        raise ValueError("retained_mode_values must have shape (n_voxels, n_modes)")
    if total_fields.shape[:3] != (incident_dirs.shape[0], 2, volume_nodes.shape[0]):
        raise ValueError("total_fields must have shape (n_pairs, 2, n_voxels, 3)")

    n_pairs = incident_dirs.shape[0]
    n_modes = phi.shape[1]
    matrix_vie = np.zeros((n_pairs, n_modes), dtype=np.complex128)
    scale = 4.0 * math.pi / (float(k) ** 2)
    comp_scale = _tensor_component_scale(tensor)

    for idx, (d, xhat) in enumerate(zip(incident_dirs, obs_dirs)):
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        projector = np.eye(3) - np.outer(xhat, xhat)
        phase = np.exp(-1j * float(k) * (volume_nodes @ xhat))
        weighted_phi = (volume_weights * phase)[:, None] * phi
        numerator = np.zeros(n_modes, dtype=np.complex128)
        denominator = 0.0 + 0.0j
        for pol_idx in range(E_basis.shape[1]):
            e = E_basis[:, pol_idx].astype(np.complex128)
            model = projector @ tensor @ e
            source_base = total_fields[idx, pol_idx] @ tensor.T
            integrals = weighted_phi.T @ source_base
            farfields = (float(k) ** 2 / (4.0 * math.pi)) * (integrals @ projector.T)
            numerator += scale * (farfields @ np.conj(model))
            denominator += np.vdot(model, model)
        if abs(denominator) > 1e-14:
            matrix_vie[idx] = numerator / denominator * comp_scale

    matrix = (
        vie_to_fourier_convention(matrix_vie)
        if conjugate_to_unified
        else matrix_vie
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size:
        positive = singular[singular > 0.0]
        condition = float(singular[0] / positive[-1]) if positive.size else math.inf
    else:
        condition = math.nan
    return BIMLinearization(
        matrix=matrix,
        matrix_norm=vector_norm(matrix),
        condition=condition,
    )


def compute_raw_bim_gpswf_linearization(
    *,
    incident_dirs: np.ndarray,
    obs_dirs: np.ndarray,
    volume_nodes: np.ndarray,
    volume_weights: np.ndarray,
    total_fields: np.ndarray,
    retained_mode_values: np.ndarray,
    tensor: np.ndarray,
    k: float,
    conjugate_to_unified: bool = False,
) -> BIMLinearization:
    """Build raw far-field channel BIM response in retained GPSWF coordinates.

    Rows are ordered exactly as ``RawVIEData.farfield_data.reshape(-1)``:
    direction pair, incident polarization, observed vector component.
    """
    del incident_dirs  # Total fields already include the incident directions.
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    volume_nodes = np.asarray(volume_nodes, dtype=float)
    volume_weights = np.asarray(volume_weights, dtype=float)
    total_fields = np.asarray(total_fields, dtype=np.complex128)
    phi = np.asarray(retained_mode_values, dtype=np.complex128)
    tensor = np.asarray(tensor, dtype=np.complex128)
    if phi.ndim != 2:
        raise ValueError("retained_mode_values must have shape (n_voxels, n_modes)")
    if total_fields.shape[:3] != (obs_dirs.shape[0], 2, volume_nodes.shape[0]):
        raise ValueError("total_fields must have shape (n_pairs, 2, n_voxels, 3)")

    n_pairs = obs_dirs.shape[0]
    n_modes = phi.shape[1]
    matrix = np.zeros((n_pairs * 6, n_modes), dtype=np.complex128)

    for idx, xhat in enumerate(obs_dirs):
        projector = np.eye(3) - np.outer(xhat, xhat)
        phase = np.exp(-1j * float(k) * (volume_nodes @ xhat))
        weighted_phi = (volume_weights * phase)[:, None] * phi
        for pol_idx in range(2):
            source_base = total_fields[idx, pol_idx] @ tensor.T
            integrals = weighted_phi.T @ source_base
            farfields = integrals @ projector.T
            if conjugate_to_unified:
                farfields = np.conj(farfields)
            row0 = idx * 6 + pol_idx * 3
            matrix[row0:row0 + 3, :] = farfields.T

    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size:
        positive = singular[singular > 0.0]
        condition = float(singular[0] / positive[-1]) if positive.size else math.inf
    else:
        condition = math.nan
    return BIMLinearization(
        matrix=matrix,
        matrix_norm=vector_norm(matrix),
        condition=condition,
    )


def solve_tikhonov_update(
    matrix: np.ndarray,
    residual: np.ndarray,
    *,
    lambda0: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve ``min ||B a-r||² + lambda ||a||²``."""
    B = np.asarray(matrix, dtype=np.complex128)
    r = np.asarray(residual, dtype=np.complex128)
    if B.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if r.shape != (B.shape[0],):
        raise ValueError("residual must have shape (matrix.shape[0],)")
    singular = np.linalg.svd(B, compute_uv=False)
    sigma_max = float(singular[0]) if singular.size else 0.0
    lam = float(lambda0) * sigma_max * sigma_max
    if B.shape[1] == 0:
        return np.zeros(0, dtype=np.complex128), {
            "lambda": lam,
            "linear_solve_residual": vector_norm(r),
            "linear_solve_rank": 0,
            "linear_solve_cond": math.nan,
            "update_norm": 0.0,
        }

    if lam > 0.0:
        augmented_matrix = np.vstack([
            B,
            math.sqrt(lam) * np.eye(B.shape[1], dtype=np.complex128),
        ])
        augmented_rhs = np.concatenate([r, np.zeros(B.shape[1], dtype=np.complex128)])
    else:
        augmented_matrix = B
        augmented_rhs = r
    coeffs, _, rank, _ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)
    linear_residual = vector_norm(B @ coeffs - r)
    condition = (
        float(singular[0] / singular[singular > 0.0][-1])
        if np.any(singular > 0.0) else math.inf
    )
    return coeffs, {
        "lambda": lam,
        "linear_solve_residual": linear_residual,
        "linear_solve_rank": int(rank),
        "linear_solve_cond": condition,
        "update_norm": vector_norm(coeffs),
    }


def _tensor_component_scale(tensor: np.ndarray) -> complex:
    coeff = tensor_coefficients_from_matrix(np.asarray(tensor, dtype=np.complex128), "isotropic")
    return complex(coeff[0])
