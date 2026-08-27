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


def forward_solver_diagnostic_summary(
    dataset: FarfieldDataset,
) -> dict[str, float | int]:
    """Return CSV-ready diagnostics specific to the forward solver."""
    metadata = dataset.metadata
    return {
        "vie_voxel_nodes": int(metadata.get("vie_voxel_nodes", 0)),
        "vie_unknowns": int(metadata.get("vie_unknowns", 0)),
        "vie_unique_incident_directions": int(
            metadata.get("vie_unique_incident_directions", 0)
        ),
        "vie_rhs_count": int(metadata.get("vie_rhs_count", 0)),
        "vie_residual_sample_count": int(
            metadata.get("vie_residual_sample_count", 0)
        ),
        "vie_linear_residual_sample_max": float(
            metadata.get("vie_linear_residual_sample_max", np.nan)
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

    ``contrast_scale`` uniformly scales the tensor contrast while preserving
    its anisotropic structure.  Remaining keyword arguments are forwarded to
    shape constructors (for example, ``cube_half_side``).
    """
    from common.phantom import (
        Block, cube_phantom, two_spheres_cube_phantom, dispersed_blocks_phantom,
    )
    from forward.vie import tensor_ball_contrast, tensor_blocks_contrast

    name = shape_name
    contrast_scale = complex(kwargs.pop("contrast_scale", 1.0))
    if name == "sphere":
        return tensor_ball_contrast(volume_nodes, 0.25, tensor,
                                    scale=contrast_scale,
                                    center=np.array([0.0, 0.0, 0.0]))
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
        return contrast_scale * Q
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
          contrast_scale * complex(b.amplitude)) for b in blocks],
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
    **shape_kwargs,
) -> FarfieldDataset:
    """Full Maxwell VIE far-field: solve total field, compute E∞.

    Direction pairs produced by the finite measurement grid often share the
    same incident direction.  The total field depends on the incident
    direction and polarization, but not on the observation direction.  We
    therefore solve the two polarization right-hand sides once per unique
    incident direction and reuse those fields for every associated observer.
    """
    from forward.vie import (
        assemble_vie_matrix, ball_voxel_grid, incident_plane_wave,
        maxwell_far_field,
    )

    incident_dirs, obs_dirs, n_p, n_geometries, geom_meta = _resolve_direction_pairs(
        p_nodes, n_geometries, incident_dirs, obs_dirs, vie_phase=True)
    volume_nodes, volume_weights, voxel_h = ball_voxel_grid(R, n_per_axis)
    tensor = reference_tensor(kind)
    Q = _build_vie_contrast(shape_name, volume_nodes, tensor, **shape_kwargs)

    A = assemble_vie_matrix(volume_nodes, volume_weights, Q, k, h=voxel_h)
    lu = lu_factor(A)

    n_meas = incident_dirs.shape[0]
    farfield_data = np.zeros((n_meas, 6), dtype=np.complex128)

    unique_incident_dirs, inverse_indices = np.unique(
        incident_dirs,
        axis=0,
        return_inverse=True,
    )
    residual_samples: list[float] = []
    residual_sample_group_limit = 3

    for incident_index, d in enumerate(unique_incident_dirs):
        measurement_indices = np.flatnonzero(inverse_indices == incident_index)
        E_basis = np.column_stack(orthonormal_basis_perp(d)).astype(np.complex128)
        rhs = np.column_stack([
            incident_plane_wave(volume_nodes, k, d, E_basis[:, col]).reshape(-1)
            for col in range(2)
        ])
        total_fields_flat = lu_solve(lu, rhs)

        if incident_index < residual_sample_group_limit:
            residual = A @ total_fields_flat - rhs
            denominator = np.maximum(np.linalg.norm(rhs, axis=0), 1e-14)
            residual_samples.extend(
                (np.linalg.norm(residual, axis=0) / denominator).astype(float).tolist()
            )

        selected_observers = obs_dirs[measurement_indices]
        for col in range(2):
            total_field = total_fields_flat[:, col].reshape((-1, 3))
            ff = maxwell_far_field(
                volume_nodes,
                volume_weights,
                Q,
                total_field,
                k,
                selected_observers,
            )
            farfield_data[measurement_indices, col * 3:(col + 1) * 3] = (
                ff / _farfield_prefactor(k)
            )

    return FarfieldDataset(
        p_nodes=p_nodes, incident_dirs=incident_dirs, obs_dirs=obs_dirs,
        farfield_data=farfield_data, data_source="full_vie",
        metadata={
            **geom_meta,
            "kind": kind,
            "k": k,
            "R": R,
            "n_per_axis": n_per_axis,
            "vie_voxel_nodes": int(volume_nodes.shape[0]),
            "vie_unknowns": int(3 * volume_nodes.shape[0]),
            "vie_unique_incident_directions": int(unique_incident_dirs.shape[0]),
            "vie_rhs_count": int(2 * unique_incident_dirs.shape[0]),
            "vie_residual_sample_count": int(len(residual_samples)),
            "vie_linear_residual_sample_max": (
                float(np.max(residual_samples)) if residual_samples else np.nan
            ),
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
    standard_noise: np.ndarray | None = None,
    prepared_farfield_data: np.ndarray | None = None,
) -> np.ndarray:
    """Polarimetric recovery: g(p) → pinv(M(p)) → c(p).

    Builds M from the dataset's explicit incident/observation direction pairs.
    For mock data, nearby measured Fourier nodes are treated as samples of the
    same target coefficient before the joint least-squares solve.

    Returns coefficients of shape ``(n_p, n_coeffs)``.
    """
    p_nodes = dataset.p_nodes
    inc = dataset.incident_dirs; obs = dataset.obs_dirs
    if prepared_farfield_data is None:
        g_raw = farfield_data_with_relative_noise(
            dataset,
            noise_level,
            rng=rng,
            standard_noise=standard_noise,
        )
    else:
        if rng is not None or standard_noise is not None:
            raise ValueError(
                "rng and standard_noise cannot be used with prepared_farfield_data"
            )
        g_raw = np.asarray(prepared_farfield_data, dtype=np.complex128)
        if g_raw.shape != dataset.farfield_data.shape:
            raise ValueError(
                "prepared_farfield_data must have the same shape as farfield_data"
            )
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


def farfield_data_with_relative_noise(
    dataset: FarfieldDataset,
    noise_level: float,
    *,
    rng: np.random.Generator | None = None,
    standard_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Return raw far-field data with node-wise relative complex noise.

    Each target Fourier node and all its polarimetric configurations form one
    channel vector.  Its noise norm is ``noise_level * ||g_i||``.  Keeping this
    operation separate lets multiple inversion/imaging methods consume exactly
    the same noisy measurement realization.
    """
    from common.utils import complex_relative_noise

    clean = np.asarray(dataset.farfield_data, dtype=np.complex128)
    noisy = clean.copy()
    n_p = int(dataset.p_nodes.shape[0])
    if n_p <= 0:
        return noisy
    if clean.shape[0] % n_p != 0:
        raise ValueError("farfield_data rows must be a multiple of p_nodes")
    if standard_noise is not None:
        standard_noise = np.asarray(standard_noise, dtype=np.complex128)
        if standard_noise.shape != clean.shape:
            raise ValueError("standard_noise must have the same shape as farfield_data")
    if noise_level > 0.0 and rng is None and standard_noise is None:
        raise ValueError("rng or standard_noise is required when noise_level is positive")

    polarimetric_J = clean.shape[0] // n_p
    for idx in range(n_p):
        rows = idx + np.arange(polarimetric_J) * n_p
        clean_vector = clean[rows].reshape(-1)
        if noise_level <= 0.0:
            continue
        if standard_noise is None:
            assert rng is not None
            noise = complex_relative_noise(clean_vector, noise_level, rng)
        else:
            eta = standard_noise[rows].reshape(-1)
            noise = (
                eta / max(float(np.linalg.norm(eta)), 1e-14)
                * float(noise_level)
                * float(np.linalg.norm(clean_vector))
            )
        noisy[rows] = (clean_vector + noise).reshape(polarimetric_J, 6)
    return noisy
