#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maxwell volume integral equation (VIE) forward solver.

Solves the frequency-domain VIE

    E(x) = E_i(x) + k² ∫_D G_k(x,y) Q(y) E(y) dy,

where ``G_k`` is the outgoing dyadic Green tensor.  The singular self-term
is included via the depolarisation dyadic for cubic voxels.

.. note::

    The far-field formulas in this module use the standard Maxwell
    convention ``exp(-i k x̂·y)``.  The resulting Born/full far-field data
    correspond to the Fourier convention ``exp(+i k (d-x̂)·y)``.
    Convert to the unified project convention ``exp(-i C p·x)`` with
    :func:`vie_to_fourier_convention`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


@dataclass(frozen=True)
class VIESolveInfo:
    """Diagnostics returned by the VIE solve."""

    gmres_info: int
    relative_residual: float
    nodes: int
    unknowns: int


# ---------------------------------------------------------------------------
# Voxel grid and contrast
# ---------------------------------------------------------------------------


def ball_voxel_grid(radius: float, n_per_axis: int) -> tuple[Array, Array, float]:
    """Return equal-volume voxel centers inside a ball.

    The cube spacing is ``h = 2 radius / n_per_axis``.  Nodes outside the
    ball are discarded, but every retained voxel keeps weight ``h³``.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if n_per_axis < 3:
        raise ValueError("n_per_axis must be at least 3")
    h = 2.0 * float(radius) / float(n_per_axis)
    axis = np.linspace(-float(radius) + 0.5 * h, float(radius) - 0.5 * h, int(n_per_axis))
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    nodes = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    mask = np.linalg.norm(nodes, axis=1) <= float(radius)
    nodes = nodes[mask]
    weights = np.full(nodes.shape[0], h**3, dtype=float)
    return nodes, weights, h


def tensor_ball_contrast(
    nodes: Array,
    radius: float,
    tensor: CArray,
    scale: float = 1.0,
    center: Array | None = None,
) -> CArray:
    """Return a constant tensor contrast inside a ball support."""
    nodes = np.asarray(nodes, dtype=float)
    tensor = np.asarray(tensor, dtype=np.complex128)
    center_vec = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n_nodes, 3)")
    if tensor.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")
    if center_vec.shape != (3,):
        raise ValueError("center must have shape (3,)")
    Q = np.zeros((nodes.shape[0], 3, 3), dtype=np.complex128)
    mask = np.linalg.norm(nodes - center_vec[None, :], axis=1) <= float(radius)
    Q[mask] = complex(scale) * tensor
    return Q


def tensor_blocks_contrast(
    nodes: Array,
    blocks: list[tuple[Array, Array, complex]],
    tensor: CArray,
) -> CArray:
    """Return a piecewise-constant tensor contrast on axis-aligned boxes.

    Each block is ``(center, half_width, amplitude)``.  This matches the
    three-block phantom used in the Born/GPSWF reconstruction experiment.
    """
    nodes = np.asarray(nodes, dtype=float)
    tensor = np.asarray(tensor, dtype=np.complex128)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n_nodes, 3)")
    if tensor.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")

    Q = np.zeros((nodes.shape[0], 3, 3), dtype=np.complex128)
    for center, half_width, amplitude in blocks:
        center_vec = np.asarray(center, dtype=float)
        half_width_vec = np.asarray(half_width, dtype=float)
        if center_vec.shape != (3,) or half_width_vec.shape != (3,):
            raise ValueError("block center and half_width must have shape (3,)")
        mask = np.all(np.abs(nodes - center_vec[None, :]) <= half_width_vec[None, :], axis=1)
        Q[mask] += complex(amplitude) * tensor
    return Q


# ---------------------------------------------------------------------------
# Incident field and dyadic Green tensor
# ---------------------------------------------------------------------------


def incident_plane_wave(
    nodes: Array, k: float, direction: Array, polarization: CArray
) -> CArray:
    """Evaluate ``E_i(x) = e exp(i k d·x)`` at volume nodes."""
    nodes = np.asarray(nodes, dtype=float)
    direction = _unit_vector(direction)
    polarization = np.asarray(polarization, dtype=np.complex128)
    if polarization.shape != (3,):
        raise ValueError("polarization must have shape (3,)")
    phase = np.exp(1j * float(k) * (nodes @ direction))
    return phase[:, None] * polarization[None, :]


def dyadic_green_tensor(r: Array, k: float) -> CArray:
    """Outgoing Maxwell dyadic Green tensor for ``r ≠ 0``.

    ``G_k(r) = (I + k^{-2} grad grad) exp(i k |r|) / (4 π |r|)``.
    """
    r = np.asarray(r, dtype=float)
    rho = float(np.linalg.norm(r))
    if rho <= 1e-14:
        return np.zeros((3, 3), dtype=np.complex128)
    khat = r / rho
    k = float(k)
    phi = np.exp(1j * k * rho) / (4.0 * math.pi * rho)
    phi_over_r = phi * (1j * k - 1.0 / rho) / rho
    phi_second = phi * (-k * k - 2j * k / rho + 2.0 / (rho * rho))
    radial_projector = np.outer(khat, khat)
    hessian = phi_second * radial_projector + phi_over_r * (np.eye(3) - radial_projector)
    return phi * np.eye(3, dtype=np.complex128) + hessian / (k * k)


# ---------------------------------------------------------------------------
# Self-term (depolarisation dyadic)
# ---------------------------------------------------------------------------


def _self_term_block(k: float, h: float, Q_i: CArray) -> CArray:
    """Self-interaction block ``K_{ii}`` for a cubic voxel of side *h*.

    The regularised self-integral of the dyadic Green tensor is

        ∫_{V_i} G_k(0, y) dy ≈ L/k² + i k V/(6π) I,

    where  ``L``  is the dimensionless depolarisation dyadic.  For a cube
    ``L = -I/3``  (the negative of the electrostatic depolarisation factor,
    which is 1/3 for each principal axis by cubic symmetry).

    Substituting into  ``K_{ii} = k² ∫ G_k dy · Q_i``  cancels the k² in
    the static part, yielding

        K_{ii} = [L + i k³ V/(6π) I] · Q_i.

    In the static limit this reproduces the Clausius-Mossotti relation
    ``E_int = 3/(ε+2) E_inc`` for a single dielectric voxel.
    """
    V = h**3
    # Static depolarisation dyadic (dimensionless, exact for cube)
    L = -np.eye(3) / 3.0
    # Radiation reaction (leading imaginary term)
    radiation = 1j * (k**3) * V / (6.0 * math.pi) * np.eye(3)
    return (L + radiation) @ Q_i


# ---------------------------------------------------------------------------
# VIE matrix assembly and solve
# ---------------------------------------------------------------------------


def assemble_vie_matrix(
    nodes: Array, weights: Array, Q: CArray, k: float, *, h: float | None = None
) -> CArray:
    """Assemble the dense collocation matrix ``A = I - K``.

    The self-term (diagonal blocks) uses the cubic depolarisation dyadic.
    ``h`` is the voxel side length; if not given it is estimated from the
    first weight as ``w_i^{1/3}``.
    """
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    Q = np.asarray(Q, dtype=np.complex128)
    n_nodes = nodes.shape[0]
    if weights.shape != (n_nodes,):
        raise ValueError("weights must have shape (n_nodes,)")
    if Q.shape != (n_nodes, 3, 3):
        raise ValueError("Q must have shape (n_nodes, 3, 3)")

    if h is None:
        h = float(np.cbrt(weights[0]))

    A = np.eye(3 * n_nodes, dtype=np.complex128)
    k2 = float(k) ** 2
    for i in range(n_nodes):
        row = slice(3 * i, 3 * i + 3)
        for j in range(n_nodes):
            col = slice(3 * j, 3 * j + 3)
            if i == j:
                block = _self_term_block(k, h, Q[j])
            else:
                block = k2 * float(weights[j]) * dyadic_green_tensor(
                    nodes[i] - nodes[j], k
                ) @ Q[j]
            A[row, col] -= block
    return A


def solve_total_field_vie(
    nodes: Array,
    weights: Array,
    Q: CArray,
    k: float,
    direction: Array,
    polarization: CArray,
    *,
    rtol: float = 1e-8,
    maxiter: int = 300,
    h: float | None = None,
) -> tuple[CArray, VIESolveInfo]:
    """Solve the dense VIE system for the total electric field."""
    E_inc = incident_plane_wave(nodes, k, direction, polarization)
    A = assemble_vie_matrix(nodes, weights, Q, k, h=h)
    b = E_inc.reshape(-1)
    solution, info = gmres(A, b, rtol=float(rtol), atol=0.0, maxiter=int(maxiter))
    residual = np.linalg.norm(A @ solution - b) / max(float(np.linalg.norm(b)), 1e-14)
    E = solution.reshape((-1, 3))
    return E, VIESolveInfo(
        gmres_info=int(info),
        relative_residual=float(residual),
        nodes=int(nodes.shape[0]),
        unknowns=int(3 * nodes.shape[0]),
    )


# ---------------------------------------------------------------------------
# Far-field computation
# ---------------------------------------------------------------------------


def maxwell_far_field(
    nodes: Array,
    weights: Array,
    Q: CArray,
    field: CArray,
    k: float,
    obs_dirs: Array,
) -> CArray:
    """Compute the Maxwell electric far field from volume current ``Q E``.

    .. math::
        E_\\infty(\\hat{x}) = \\frac{k^2}{4\\pi}
        P_{\\hat{x}} \\int \\exp(-i k \\hat{x}\\cdot y) Q(y) E(y) dy.
    """
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    Q = np.asarray(Q, dtype=np.complex128)
    field = np.asarray(field, dtype=np.complex128)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if obs_dirs.ndim != 2 or obs_dirs.shape[1] != 3:
        raise ValueError("obs_dirs must have shape (n_obs, 3)")
    source = np.einsum("nij,nj->ni", Q, field)
    farfield = np.zeros((obs_dirs.shape[0], 3), dtype=np.complex128)
    for idx, obs in enumerate(obs_dirs):
        xhat = _unit_vector(obs)
        phase = np.exp(-1j * float(k) * (nodes @ xhat))
        integral = np.sum((weights * phase)[:, None] * source, axis=0)
        projector = np.eye(3) - np.outer(xhat, xhat)
        farfield[idx] = (float(k) ** 2 / (4.0 * math.pi)) * (projector @ integral)
    return farfield


def maxwell_born_far_field(
    nodes: Array,
    weights: Array,
    Q: CArray,
    k: float,
    direction: Array,
    polarization: CArray,
    obs_dirs: Array,
) -> CArray:
    """Born far field: replace total field with incident field.

    The resulting data carries the Fourier phase ``exp(+i k (d - x̂)·y)``
    (opposite to the unified project convention).  Convert with
    :func:`vie_to_fourier_convention` before feeding into GPSWF inversion.
    """
    E_inc = incident_plane_wave(nodes, k, direction, polarization)
    return maxwell_far_field(nodes, weights, Q, E_inc, k, obs_dirs)


# ---------------------------------------------------------------------------
# Convention conversion
# ---------------------------------------------------------------------------


def vie_to_fourier_convention(data: CArray) -> CArray:
    """Convert VIE far-field data to the unified Fourier convention.

    VIE far-field data carries the phase ``exp(+i k (d - x̂)·y)``
    (= ``exp(+i C p·y)``).  The unified project convention is
    ``exp(-i C p·y)``.  The conversion is simply complex conjugation.

    Apply this to any scalar Fourier data extracted from
    :func:`maxwell_born_far_field` or :func:`maxwell_far_field` before
    passing it to the GPSWF reconstruction pipeline.
    """
    return np.conj(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(v: Array) -> Array:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError("vector must have shape (3,)")
    norm = float(np.linalg.norm(v))
    if norm <= 1e-14:
        raise ValueError("vector must be nonzero")
    return v / norm
