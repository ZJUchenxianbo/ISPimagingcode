#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified far-field data generation and polarimetric recovery.

Layer 1 — forward data generation:
  - analytic_born_farfield_dataset: analytical Born far-field
  - discrete_vie_born_farfield_dataset: VIE discretised Born
  - full_vie_farfield_dataset: Full Maxwell VIE

Layer 2 — inversion entry:
  - farfield_dataset_to_qhat: polarimetric recovery → Q̂(p)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from common.quadrature import (
    admissible_farfield_pairs_from_nodes,
    orthonormal_basis_perp,
    paired_farfield_fourier_nodes,
)
from common.phantom import (
    Block,
    TensorKind,
    block_fourier_profile,
    reference_tensor,
    tensor_basis,
    tensor_coefficients_from_matrix,
)
from common.polarimetric import build_polarimetric_matrix_from_directions


@dataclass
class FarfieldDataset:
    """Far-field measurement data for polarimetric inversion."""

    p_nodes: np.ndarray                # (n_p, 3)  Fourier ball nodes
    incident_dirs: np.ndarray          # (n_meas, 3)  incident directions
    obs_dirs: np.ndarray               # (n_meas, 3)  observation directions
    farfield_data: np.ndarray          # (n_meas, 6)  2-pol × 3-comp stacked
    data_source: str                   # "analytic_born" | "vie_born" | "full_vie"
    metadata: dict = field(default_factory=dict)


def polarimetric_diagnostic_summary(dataset: FarfieldDataset) -> dict[str, float | int]:
    """Return CSV-ready stability diagnostics after polarimetric recovery."""
    metadata = dataset.metadata
    return {
        "polarimetric_J": int(metadata.get("polarimetric_J", 0)),
        "polarimetric_rank_min": int(metadata.get("polarimetric_rank_min", 0)),
        "polarimetric_sigma_min_min": float(
            metadata.get("polarimetric_sigma_min_min", np.nan)
        ),
        "polarimetric_sigma_min_median": float(
            metadata.get("polarimetric_sigma_min_median", np.nan)
        ),
        "polarimetric_condition_median": float(
            metadata.get("polarimetric_condition_median", np.nan)
        ),
        "polarimetric_condition_max": float(
            metadata.get("polarimetric_condition_max", np.nan)
        ),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _admissible_geometries(p_nodes: np.ndarray, n_geometries: int = 6):
    inc_list, obs_list = [], []
    for b in range(n_geometries):
        inc, obs, _ = admissible_farfield_pairs_from_nodes(
            p_nodes, branch_index=b, branch_count=n_geometries)
        inc_list.append(inc); obs_list.append(obs)
    n_p = p_nodes.shape[0]
    return (np.concatenate(inc_list, axis=0), np.concatenate(obs_list, axis=0), n_p, n_geometries)


def _resolve_direction_pairs(
    p_nodes: np.ndarray,
    n_geometries: int,
    incident_dirs: np.ndarray | None,
    obs_dirs: np.ndarray | None,
    *,
    vie_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, dict]:
    """Return direction pairs ordered by branch blocks.

    ``p_nodes`` are always the reconstruction Fourier nodes.  For VIE data,
    physical direction pairs are constructed for ``-p_nodes`` by default so
    that the raw VIE phase ``exp(+i k(d-xhat).x)`` matches the project
    convention ``exp(-i C p.x)``.
    """
    p_nodes = np.asarray(p_nodes, dtype=float)
    if p_nodes.ndim != 2 or p_nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if (incident_dirs is None) != (obs_dirs is None):
        raise ValueError("incident_dirs and obs_dirs must be provided together")

    n_p = p_nodes.shape[0]
    if incident_dirs is None:
        physical_nodes = -p_nodes if vie_phase else p_nodes
        inc, obs, n_p, n_geometries = _admissible_geometries(physical_nodes, n_geometries)
        physical_sign = -1 if vie_phase else 1
    else:
        inc = np.asarray(incident_dirs, dtype=float)
        obs = np.asarray(obs_dirs, dtype=float)
        if inc.shape != obs.shape or inc.ndim != 2 or inc.shape[1] != 3:
            raise ValueError("incident_dirs and obs_dirs must both have shape (n_meas, 3)")
        if inc.shape[0] % max(n_p, 1) != 0:
            raise ValueError("number of direction pairs must be a multiple of p_nodes")
        n_geometries = inc.shape[0] // max(n_p, 1)
        paired_nodes = paired_farfield_fourier_nodes(inc, obs).reshape(n_geometries, n_p, 3)
        err_plus = float(np.max(np.linalg.norm(paired_nodes - p_nodes[None, :, :], axis=2)))
        err_minus = float(np.max(np.linalg.norm(paired_nodes + p_nodes[None, :, :], axis=2)))
        physical_sign = -1 if err_minus <= err_plus else 1

    metadata = {
        "n_geometries": int(n_geometries),
        "physical_node_sign": int(physical_sign),
        "fourier_convention": "exp(-i C p.x)",
    }
    return inc, obs, n_p, int(n_geometries), metadata


def _farfield_prefactor(k: float) -> float:
    return float(k) ** 2 / (4.0 * np.pi)


def _build_vie_contrast(shape_name: str, volume_nodes, tensor, **kwargs):
    """Build Q on voxel grid for any shape (matches figure3 logic).

    Extra kwargs are forwarded to shape constructors (e.g. ``cube_half_side``).
    """
    from common.phantom import (
        Block, cube_phantom, two_spheres_cube_phantom, dispersed_blocks_phantom,
    )
    from forward.vie import tensor_ball_contrast, tensor_blocks_contrast

    name = shape_name
    if name == "sphere":
        return tensor_ball_contrast(volume_nodes, 0.25, tensor,
                                    scale=1.0, center=np.array([0.0, 0.0, 0.0]))
    elif name == "inhomogeneous":
        Q = np.zeros((volume_nodes.shape[0], 3, 3), dtype=np.complex128)
        bumps = [
            (np.array([-0.20, 0.15, 0.0]), 0.12, 1.0 + 0.0j),
            (np.array([0.25, 0.10, 0.0]), 0.08, 0.7 + 0.2j),
            (np.array([0.05, -0.25, 0.0]), 0.14, 1.2 - 0.1j),
        ]
        for center, sigma, amp in bumps:
            r_sq = np.sum((volume_nodes - center[None, :])**2, axis=1)
            scalar = complex(amp) * np.exp(-0.5 * r_sq / sigma**2)
            for a in range(3):
                Q[:, a, a] += scalar
        return Q
    elif name == "cube":
        half_side = float(kwargs.get("cube_half_side", 0.2))
        blocks = cube_phantom(center=(0.0, 0.0, 0.0), half_side=half_side, amplitude=1.0 + 0.0j)
    elif name == "two_spheres_cube":
        blocks = two_spheres_cube_phantom()
    elif name == "dispersed":
        blocks = dispersed_blocks_phantom()
    elif name == "three_blocks":
        from common.phantom import three_block_phantom
        blocks = three_block_phantom("born")
    else:
        raise ValueError(f"Unknown shape: {name!r}")
    return tensor_blocks_contrast(
        volume_nodes,
        [(np.asarray(b.center, dtype=float), np.asarray(b.half_width, dtype=float),
          complex(b.amplitude)) for b in blocks],
        tensor)


# ---------------------------------------------------------------------------
# Layer 1: forward data generators
# ---------------------------------------------------------------------------


def analytic_born_farfield_dataset(
    shape_name: str,
    p_nodes: np.ndarray,
    kind: TensorKind = "full",
    k: float = 15.0,
    *,
    n_geometries: int = 6,
    incident_dirs: np.ndarray | None = None,
    obs_dirs: np.ndarray | None = None,
) -> FarfieldDataset:
    """Analytical Born far-field: compute Q̂(ξ) analytically, build g = M(p)c(p).

    If ``incident_dirs`` / ``obs_dirs`` are provided, they are used directly
    (e.g. from mock-quadrature matching).  Otherwise admissible direction
    pairs are constructed from ``p_nodes``.
    """
    from common.phantom import _shape_truth_and_fourier

    incident_dirs, obs_dirs, n_p, n_geometries, geom_meta = _resolve_direction_pairs(
        p_nodes, n_geometries, incident_dirs, obs_dirs, vie_phase=False)
    n_meas = incident_dirs.shape[0]

    # Analytical Fourier Q̂ at each p (scalar)
    C = 2.0 * k
    _, _, _, scalar_fourier = _shape_truth_and_fourier(shape_name, p_nodes, grid_size=1, C=C)
    coeff0 = tensor_coefficients_from_matrix(reference_tensor(kind), kind)
    coeffs_per_p = scalar_fourier[:, None] * coeff0[None, :]  # (n_p, n_coeffs)

    basis = tensor_basis(kind)
    farfield_data = np.zeros((n_meas, 6), dtype=np.complex128)

    for j in range(n_meas):
        d = incident_dirs[j]; xhat = obs_dirs[j]
        idx = j % n_p  # which p node
        P = np.eye(3) - np.outer(xhat, xhat)
        Qhat = sum(coeffs_per_p[idx, r] * basis[r] for r in range(len(basis)))
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        ff = P @ Qhat @ E_basis  # g = M(p)c(p), no k²/(4π) prefactor
        farfield_data[j] = ff.reshape(-1, order='F')

    return FarfieldDataset(
        p_nodes=p_nodes, incident_dirs=incident_dirs, obs_dirs=obs_dirs,
        farfield_data=farfield_data, data_source="analytic_born",
        metadata={
            **geom_meta,
            "kind": kind,
            "k": k,
            "farfield_normalization": "M_c",
            "prefactor_removed": 1.0,
        },
    )


def analytic_block_born_farfield_dataset(
    blocks: list[Block],
    target_p_nodes: np.ndarray,
    kind: TensorKind = "full",
    k: float = 15.0,
    *,
    incident_dirs: np.ndarray,
    obs_dirs: np.ndarray,
) -> FarfieldDataset:
    """Analytical Born far-field for a piecewise-constant block phantom.

    ``target_p_nodes`` are the Fourier-ball nodes used by the inverse
    quadrature.  The far-field values themselves are evaluated at the actual
    Fourier nodes associated with each supplied direction pair.  Consequently,
    mock-node mismatch remains present in the generated data.

    The VIE convention gives ``exp(+i k(d-xhat).x)``.  Under the project
    convention ``exp(-i C p.x)``, ``C=2k``, the Fourier node used to evaluate
    the block profile is therefore ``p=(xhat-d)/2``.
    """
    target_p_nodes = np.asarray(target_p_nodes, dtype=float)
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if target_p_nodes.ndim != 2 or target_p_nodes.shape[1] != 3:
        raise ValueError("target_p_nodes must have shape (n_nodes, 3)")
    if incident_dirs.shape != obs_dirs.shape or incident_dirs.ndim != 2 or incident_dirs.shape[1] != 3:
        raise ValueError("incident_dirs and obs_dirs must both have shape (n_meas, 3)")

    n_p = target_p_nodes.shape[0]
    n_meas = incident_dirs.shape[0]
    if n_p <= 0 or n_meas % n_p != 0:
        raise ValueError("number of direction pairs must be a multiple of target_p_nodes")

    physical_nodes = paired_farfield_fourier_nodes(incident_dirs, obs_dirs)
    actual_fourier_nodes = -physical_nodes
    scalar_fourier = block_fourier_profile(actual_fourier_nodes, blocks, C=2.0 * float(k))
    tensor = reference_tensor(kind)
    farfield_data = np.zeros((n_meas, 6), dtype=np.complex128)

    for measurement_index, (d, xhat) in enumerate(zip(incident_dirs, obs_dirs)):
        projector = np.eye(3) - np.outer(xhat, xhat)
        incident_polarizations = np.column_stack(orthonormal_basis_perp(d))
        qhat = scalar_fourier[measurement_index] * tensor
        farfield = projector @ qhat @ incident_polarizations
        farfield_data[measurement_index] = farfield.reshape(-1, order="F")

    return FarfieldDataset(
        p_nodes=target_p_nodes,
        incident_dirs=incident_dirs,
        obs_dirs=obs_dirs,
        farfield_data=farfield_data,
        data_source="analytic_block_born",
        metadata={
            "kind": kind,
            "k": float(k),
            "n_geometries": int(n_meas // n_p),
            "physical_node_sign": -1,
            "fourier_convention": "exp(-i C p.x)",
            "farfield_normalization": "M_c",
            "prefactor_removed": 1.0,
        },
    )


def discrete_vie_born_farfield_dataset(
    shape_name: str,
    p_nodes: np.ndarray,
    kind: TensorKind = "full",
    k: float = 15.0,
    R: float = 1.0,
    n_per_axis: int = 11,
    n_geometries: int = 6,
    *,
    incident_dirs: np.ndarray | None = None,
    obs_dirs: np.ndarray | None = None,
    **shape_kwargs,
) -> FarfieldDataset:
    """VIE-discretised Born far-field: voxelised phantom, E = E_inc.

    Extra keyword arguments are forwarded to ``_build_vie_contrast``
    (e.g. ``cube_half_side=0.4``).
    """
    from forward.vie import ball_voxel_grid, maxwell_born_far_field

    incident_dirs, obs_dirs, n_p, n_geometries, geom_meta = _resolve_direction_pairs(
        p_nodes, n_geometries, incident_dirs, obs_dirs, vie_phase=True)
    volume_nodes, volume_weights, _ = ball_voxel_grid(R, n_per_axis)
    tensor = reference_tensor(kind)
    Q = _build_vie_contrast(shape_name, volume_nodes, tensor, **shape_kwargs)

    n_meas = incident_dirs.shape[0]
    farfield_data = np.zeros((n_meas, 6), dtype=np.complex128)

    for j in range(n_meas):
        d = incident_dirs[j]; xhat = obs_dirs[j]
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        for col in range(2):
            e = E_basis[:, col].astype(np.complex128)
            ff = maxwell_born_far_field(volume_nodes, volume_weights, Q, k, d, e, xhat[None, :])[0]
            farfield_data[j, col * 3:(col + 1) * 3] = ff / _farfield_prefactor(k)

    return FarfieldDataset(
        p_nodes=p_nodes, incident_dirs=incident_dirs, obs_dirs=obs_dirs,
        farfield_data=farfield_data, data_source="vie_born",
        metadata={
            **geom_meta,
            "kind": kind,
            "k": k,
            "R": R,
            "n_per_axis": n_per_axis,
            "farfield_normalization": "M_c",
            "prefactor_removed": _farfield_prefactor(k),
        },
    )


def full_vie_farfield_dataset(
    shape_name: str,
    p_nodes: np.ndarray,
    kind: TensorKind = "full",
    k: float = 15.0,
    R: float = 1.0,
    n_per_axis: int = 11,
    n_geometries: int = 6,
    *,
    incident_dirs: np.ndarray | None = None,
    obs_dirs: np.ndarray | None = None,
) -> FarfieldDataset:
    """Full Maxwell VIE far-field: solve total field, compute E∞."""
    from forward.vie import (
        assemble_vie_matrix, ball_voxel_grid, incident_plane_wave,
        maxwell_far_field,
    )

    incident_dirs, obs_dirs, n_p, n_geometries, geom_meta = _resolve_direction_pairs(
        p_nodes, n_geometries, incident_dirs, obs_dirs, vie_phase=True)
    volume_nodes, volume_weights, voxel_h = ball_voxel_grid(R, n_per_axis)
    tensor = reference_tensor(kind)
    Q = _build_vie_contrast(shape_name, volume_nodes, tensor)

    A = assemble_vie_matrix(volume_nodes, volume_weights, Q, k, h=voxel_h)
    lu = lu_factor(A)

    n_meas = incident_dirs.shape[0]
    farfield_data = np.zeros((n_meas, 6), dtype=np.complex128)

    for j in range(n_meas):
        d = incident_dirs[j]; xhat = obs_dirs[j]
        E_basis = np.column_stack(orthonormal_basis_perp(d))
        for col in range(2):
            e = E_basis[:, col].astype(np.complex128)
            rhs = incident_plane_wave(volume_nodes, k, d, e).reshape(-1)
            total_field = lu_solve(lu, rhs).reshape((-1, 3))
            ff = maxwell_far_field(volume_nodes, volume_weights, Q, total_field, k, xhat[None, :])[0]
            farfield_data[j, col * 3:(col + 1) * 3] = ff / _farfield_prefactor(k)

    return FarfieldDataset(
        p_nodes=p_nodes, incident_dirs=incident_dirs, obs_dirs=obs_dirs,
        farfield_data=farfield_data, data_source="full_vie",
        metadata={
            **geom_meta,
            "kind": kind,
            "k": k,
            "R": R,
            "n_per_axis": n_per_axis,
            "farfield_normalization": "M_c",
            "prefactor_removed": _farfield_prefactor(k),
        },
    )


# ---------------------------------------------------------------------------
# Layer 2: unified inversion entry
# ---------------------------------------------------------------------------


def farfield_dataset_to_qhat(
    dataset: FarfieldDataset,
    kind: TensorKind = "full",
    *,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Polarimetric recovery: g(p) → pinv(M(p)) → c(p).

    Builds M from the dataset's explicit incident/observation direction pairs.
    For mock data, nearby measured Fourier nodes are treated as samples of the
    same target coefficient before the joint least-squares solve.

    Returns coefficients of shape ``(n_p, n_coeffs)``.
    """
    from common.utils import complex_relative_noise

    p_nodes = dataset.p_nodes
    inc = dataset.incident_dirs; obs = dataset.obs_dirs; g_raw = dataset.farfield_data
    n_p = p_nodes.shape[0]
    n_meas = g_raw.shape[0]
    if n_p <= 0:
        return np.zeros((0, 0), dtype=np.complex128)
    if n_meas % n_p != 0:
        raise ValueError("farfield_data rows must be a multiple of p_nodes")
    polarimetric_J = max(1, n_meas // n_p)
    basis = tensor_basis(kind)
    n_coeffs = len(basis)
    recovered = np.zeros((n_p, n_coeffs), dtype=np.complex128)
    ranks = np.empty(n_p, dtype=float)
    sigma_min = np.empty(n_p, dtype=float)
    condition_numbers = np.empty(n_p, dtype=float)

    if noise_level > 0.0 and rng is None:
        raise ValueError("rng is required when noise_level is positive")

    for idx in range(n_p):
        rows = idx + np.arange(polarimetric_J) * n_p
        M = build_polarimetric_matrix_from_directions(inc[rows], obs[rows], kind)
        g_vec = g_raw[rows].reshape(-1)

        singular_values = np.linalg.svd(M, compute_uv=False)
        rank = int(np.linalg.matrix_rank(M))
        ranks[idx] = rank
        sigma_min[idx] = float(singular_values[-1])
        condition_numbers[idx] = float(
            singular_values[0] / max(singular_values[-1], 1e-14)
        )
        if rank < n_coeffs:
            raise ValueError(
                "polarimetric recovery is rank deficient at target "
                f"{idx}: matrix_shape={M.shape}, rank={rank}, "
                f"required={n_coeffs}, polarimetric_J={polarimetric_J}"
            )

        if noise_level > 0.0:
            g_vec = g_vec + complex_relative_noise(g_vec, noise_level, rng)
        recovered[idx] = np.linalg.pinv(M) @ g_vec

    dataset.metadata.update({
        "polarimetric_J": int(polarimetric_J),
        "polarimetric_ranks": ranks,
        "polarimetric_rank_min": int(np.min(ranks)),
        "polarimetric_sigma_min": sigma_min,
        "polarimetric_sigma_min_min": float(np.min(sigma_min)),
        "polarimetric_sigma_min_median": float(np.median(sigma_min)),
        "polarimetric_condition_numbers": condition_numbers,
        "polarimetric_condition_median": float(np.median(condition_numbers)),
        "polarimetric_condition_max": float(np.max(condition_numbers)),
        "farfield_noise_level": float(noise_level),
    })

    return recovered
