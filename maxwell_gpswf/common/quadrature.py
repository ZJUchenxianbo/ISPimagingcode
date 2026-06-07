#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quadrature nodes on the sphere and inside the Fourier ball.

Maxwell Section 6 uses a tensor product of radial Gauss-Jacobi nodes and
Lebedev (or other) angular quadrature on S².
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import lebedev_rule
from scipy.special import roots_jacobi
from scipy.spatial import cKDTree

Array = NDArray[np.float64]
SphereRule = Literal["lebedev", "fibonacci", "rings"]

LEBEDEV_ORDERS_AND_COUNTS: tuple[tuple[int, int], ...] = (
    (3, 6), (5, 14), (7, 26), (9, 38), (11, 50), (13, 74), (15, 86),
    (17, 110), (19, 146), (21, 170), (23, 194), (25, 230), (27, 266),
    (29, 302), (31, 350), (35, 434), (41, 590), (47, 770), (53, 974),
    (59, 1202), (65, 1454), (71, 1730), (77, 2030), (83, 2354),
    (89, 2702), (95, 3074), (101, 3470), (107, 3890), (113, 4334),
    (119, 4802), (125, 5294), (131, 5810),
)


# ---------------------------------------------------------------------------
# Sphere quadrature rules
# ---------------------------------------------------------------------------


def fibonacci_sphere_directions(n_dirs: int) -> Array:
    """Deterministic near-uniform directions on the unit sphere."""
    if n_dirs <= 0:
        raise ValueError("n_dirs must be positive")
    idx = np.arange(n_dirs, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_dirs)
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = golden_angle * idx
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def equal_area_sphere_directions(n_dirs: int) -> Array:
    """Deterministic equal-area ring directions on the unit sphere."""
    if n_dirs <= 0:
        raise ValueError("n_dirs must be positive")
    if n_dirs == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    n_rings = max(2, int(round(math.sqrt(float(n_dirs) / 2.0))))
    z_centers = 1.0 - 2.0 * (np.arange(n_rings, dtype=float) + 0.5) / float(n_rings)
    ring_weights = np.sqrt(np.maximum(1.0 - z_centers * z_centers, 0.0))
    ring_weights = ring_weights / max(float(np.sum(ring_weights)), 1e-14)
    raw_counts = ring_weights * float(n_dirs)
    counts = np.maximum(1, np.floor(raw_counts).astype(int))

    while int(np.sum(counts)) < n_dirs:
        deficits = raw_counts - counts
        counts[int(np.argmax(deficits))] += 1
    while int(np.sum(counts)) > n_dirs:
        candidates = np.where(counts > 1)[0]
        excess = counts[candidates] - raw_counts[candidates]
        counts[int(candidates[int(np.argmax(excess))])] -= 1

    points = []
    for ring_idx, (z, count) in enumerate(zip(z_centers, counts)):
        radius = math.sqrt(max(1.0 - float(z) * float(z), 0.0))
        offset = math.pi / float(count) if ring_idx % 2 else 0.0
        phi = offset + 2.0 * math.pi * np.arange(int(count), dtype=float) / float(count)
        points.append(np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.full(int(count), z)]))
    return np.vstack(points)


def lebedev_sphere_quadrature(min_points: int) -> tuple[Array, Array, int]:
    """Return Lebedev angular nodes with at least ``min_points`` directions.

    Prefers positive-weight rules for GPSWF projection diagnostics.
    """
    if min_points <= 0:
        raise ValueError("min_points must be positive")
    chosen: tuple[Array, Array, int] | None = None
    for candidate_order, count in LEBEDEV_ORDERS_AND_COUNTS:
        if count >= int(min_points):
            points, weights = lebedev_rule(candidate_order)
            weights = np.asarray(weights, dtype=float)
            if float(np.min(weights)) <= 0.0:
                continue
            chosen = (np.asarray(points.T, dtype=float), weights, int(candidate_order))
            break
    if chosen is None:
        raise ValueError(
            f"requested at least {min_points} positive-weight Lebedev points, "
            f"but no supported rule was found"
        )
    return chosen


def sphere_quadrature(
    n_dirs: int, rule: SphereRule = "lebedev"
) -> tuple[Array, Array, str]:
    """Return angular quadrature nodes and weights on the unit sphere."""
    if rule == "lebedev":
        directions, weights, order = lebedev_sphere_quadrature(n_dirs)
        return directions, weights, f"lebedev_order_{order}"
    if rule == "fibonacci":
        directions = fibonacci_sphere_directions(n_dirs)
    elif rule == "rings":
        directions = equal_area_sphere_directions(n_dirs)
    else:
        raise ValueError("rule must be 'lebedev', 'fibonacci', or 'rings'")
    weights = np.full(directions.shape[0], 4.0 * math.pi / float(directions.shape[0]), dtype=float)
    return directions, weights, rule


# ---------------------------------------------------------------------------
# Fourier-ball node mapping
# ---------------------------------------------------------------------------


def farfield_fourier_nodes(incident_dirs: Array, obs_dirs: Array) -> Array:
    """Map far-field direction pairs to normalized Fourier ball nodes.

    ``p = (d - xhat) / 2`` — the 3D analogue of the disk mapping from the
    scalar low-rank paper.
    """
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if incident_dirs.ndim != 2 or incident_dirs.shape[1] != 3:
        raise ValueError("incident_dirs must have shape (n_inc, 3)")
    if obs_dirs.ndim != 2 or obs_dirs.shape[1] != 3:
        raise ValueError("obs_dirs must have shape (n_obs, 3)")
    return 0.5 * (incident_dirs[:, None, :] - obs_dirs[None, :, :]).reshape(-1, 3)


def paired_farfield_fourier_nodes(incident_dirs: Array, obs_dirs: Array) -> Array:
    """Map paired far-field directions to Fourier ball nodes (one-to-one)."""
    incident_dirs = np.asarray(incident_dirs, dtype=float)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if incident_dirs.shape != obs_dirs.shape or incident_dirs.ndim != 2 or incident_dirs.shape[1] != 3:
        raise ValueError("incident_dirs and obs_dirs must both have shape (n_pairs, 3)")
    return 0.5 * (incident_dirs - obs_dirs)


def interior_ball_nodes(nodes: Array, radius_tol: float = 1e-12) -> Array:
    """Keep Fourier ball nodes satisfying ``|p| < 1``."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n_nodes, 3)")
    radii = np.linalg.norm(nodes, axis=1)
    return nodes[radii < 1.0 - float(radius_tol)]


# ---------------------------------------------------------------------------
# Ball quadrature (Section 6)
# ---------------------------------------------------------------------------


def ball_quadrature_nodes(
    n_radial: int,
    n_dirs: int,
    *,
    angular_rule: SphereRule = "lebedev",
) -> tuple[Array, Array, Array]:
    """Build Section 6 tensor-product quadrature nodes in the unit ball.

    Radial: Gauss-Jacobi with weight ``(1+η)^{1/2}``, ``η = 2r² - 1``,
    with factor ``1/(4√2)``.

    Angular: defaults to Lebedev nodes on S².  ``n_dirs`` is the *minimum*
    requested number of angular nodes; the actual count comes from the
    nearest positive-weight Lebedev rule.
    """
    if n_radial <= 0 or n_dirs <= 0:
        raise ValueError("n_radial and n_dirs must be positive")
    eta, radial_raw_weights = roots_jacobi(n_radial, 0.0, 0.5)
    radii = np.sqrt(0.5 * (1.0 + eta))
    radial_weights = radial_raw_weights / (4.0 * math.sqrt(2.0))
    directions, angular_weights, _ = sphere_quadrature(n_dirs, angular_rule)
    nodes = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    weights = (radial_weights[:, None] * angular_weights[None, :]).reshape(-1)
    return nodes, weights, radii


# ---------------------------------------------------------------------------
# Mock-quadrature matching
# ---------------------------------------------------------------------------


def match_mock_quadrature_nodes(
    target_nodes: Array, available_nodes: Array
) -> tuple[Array, Array]:
    """Find nearest available far-field node for each target quadrature node."""
    target_nodes = np.asarray(target_nodes, dtype=float)
    available_nodes = np.asarray(available_nodes, dtype=float)
    if target_nodes.ndim != 2 or target_nodes.shape[1] != 3:
        raise ValueError("target_nodes must have shape (n_target, 3)")
    if available_nodes.ndim != 2 or available_nodes.shape[1] != 3:
        raise ValueError("available_nodes must have shape (n_available, 3)")
    tree = cKDTree(available_nodes)
    distances, indices = tree.query(target_nodes, k=1)
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=float)


# ---------------------------------------------------------------------------
# Direction-pair geometry helpers
# ---------------------------------------------------------------------------


def orthonormal_basis_perp(v: Array) -> tuple[Array, Array]:
    """Build two real orthonormal vectors spanning ``v`` perpendicular."""
    from common.utils import vector_norm

    v = np.asarray(v, dtype=float)
    nv = vector_norm(v)
    if nv <= 1e-14:
        raise ValueError("v must be nonzero")
    v_unit = v / nv
    axes = np.eye(3)
    tmp = axes[int(np.argmin(np.abs(axes @ v_unit)))]
    a = tmp - float(tmp @ v_unit) * v_unit
    a /= vector_norm(a)
    b = np.cross(v_unit, a)
    b /= vector_norm(b)
    return a, b


def admissible_farfield_pairs_from_nodes(
    p_nodes: Array, branch_index: int = 0, branch_count: int = 3
) -> tuple[Array, Array, Array]:
    """Construct Maxwell-admissible direction pairs for prescribed ball nodes.

    For ``|p| < 1``, choose ``s ⊥ p`` and set
    ``d = p + √(1-|p|²) s``, ``x̂ = -p + √(1-|p|²) s``.
    Then ``p = (d - x̂)/2`` exactly.
    """
    from common.utils import vector_norm

    p_nodes = np.asarray(p_nodes, dtype=float)
    if p_nodes.ndim != 2 or p_nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if branch_count <= 0:
        raise ValueError("branch_count must be positive")
    branch_index = int(branch_index) % int(branch_count)
    incident = np.zeros_like(p_nodes)
    obs = np.zeros_like(p_nodes)
    for idx, p in enumerate(p_nodes):
        rho = vector_norm(p)
        if rho >= 1.0 + 1e-12:
            raise ValueError("all p_nodes must satisfy |p| <= 1")
        if rho > 1e-14:
            u, v = orthonormal_basis_perp(p)
        else:
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
        angle = 2.0 * math.pi * float(branch_index) / float(branch_count)
        s = math.cos(angle) * u + math.sin(angle) * v
        transverse_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
        d = p + transverse_scale * s
        xhat = -p + transverse_scale * s
        incident[idx] = d / vector_norm(d)
        obs[idx] = xhat / vector_norm(xhat)
    return incident, obs, paired_farfield_fourier_nodes(incident, obs)


def generate_data_nodes(
    target_nodes: Array,
    requested_measure_dirs: int,
    *,
    data_mode: str = "mock",
    branch_count: int = 3,
    angular_rule: SphereRule = "lebedev",
) -> tuple[Array, Array, Array, Array, dict]:
    """Generate far-field data nodes from target quadrature nodes.

    Two modes:

    ``"mock"`` (06005-style)
        Finite incident/observation directions → measured nodes
        ``p = (d-x̂)/2`` → nearest measured node for each target node.
        Returns mock distances as a quality diagnostic.

    ``"ideal"``
        For each target node *p*, construct admissible direction pairs
        via :func:`admissible_farfield_pairs_from_nodes` so that
        ``p = (d-x̂)/2`` exactly.  No mock-quadrature error.

    Returns
    -------
    p_nodes : (n_target, 3)
        Fourier nodes where data is evaluated.
    incident_dirs : (n_target, 3)
    obs_dirs : (n_target, 3)
    mock_distances : (n_target,)
        Zero for ``"ideal"`` mode.
    info : dict
        ``n_measure_dirs``, ``measure_rule``, ``available_nodes`` (mock only).
    """
    target_nodes = np.asarray(target_nodes, dtype=float)

    if data_mode == "ideal":
        n_target = target_nodes.shape[0]
        # Use multiple branches to get enough geometries per node
        incident_list, obs_list = [], []
        branches_per_node = max(1, branch_count)
        for b in range(branches_per_node):
            inc, obs, _ = admissible_farfield_pairs_from_nodes(
                target_nodes, branch_index=b, branch_count=branches_per_node)
            incident_list.append(inc)
            obs_list.append(obs)
        incident_dirs = np.concatenate(incident_list, axis=0)
        obs_dirs = np.concatenate(obs_list, axis=0)
        # Each target node appears branch_count times
        p_nodes = np.tile(target_nodes, (branches_per_node, 1))
        mock_distances = np.zeros(p_nodes.shape[0], dtype=float)
        info = {"n_measure_dirs": branches_per_node * target_nodes.shape[0],
                "measure_rule": "ideal_admissible",
                "data_mode": "ideal"}
        return p_nodes, incident_dirs, obs_dirs, mock_distances, info

    elif data_mode == "mock":
        directions, _, measure_rule = sphere_quadrature(requested_measure_dirs, angular_rule)
        n_measure_dirs = directions.shape[0]
        raw_available = farfield_fourier_nodes(directions, directions)
        available_nodes = interior_ball_nodes(raw_available)
        indices, distances = match_mock_quadrature_nodes(target_nodes, available_nodes)
        p_nodes = available_nodes[indices]
        # Extract matched direction pairs
        interior_mask = np.linalg.norm(raw_available, axis=1) < 1.0 - 1e-12
        inc_idx = np.repeat(np.arange(n_measure_dirs), n_measure_dirs)[interior_mask]
        obs_idx = np.tile(np.arange(n_measure_dirs), n_measure_dirs)[interior_mask]
        incident_dirs = directions[inc_idx[indices]]
        obs_dirs = directions[obs_idx[indices]]
        info = {"n_measure_dirs": int(n_measure_dirs),
                "measure_rule": measure_rule,
                "available_nodes": int(available_nodes.shape[0]),
                "data_mode": "mock"}
        return p_nodes, incident_dirs, obs_dirs, distances, info

    else:
        raise ValueError(f"Unknown data_mode: {data_mode!r}")


def build_geometries_from_p(
    p: Array, J: int = 6
) -> list[tuple[Array, Array, Array]]:
    """Construct deterministic Maxwell-admissible geometries for one p."""
    from common.utils import vector_norm

    p = np.asarray(p, dtype=float)
    rho = vector_norm(p)
    if rho >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if J <= 0:
        raise ValueError("J must be positive")

    if rho > 1e-14:
        u, v = orthonormal_basis_perp(p)
    else:
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])

    transverse_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
    geometries: list[tuple[Array, Array, Array]] = []
    for j in range(J):
        angle = 2.0 * math.pi * j / J
        s = math.cos(angle) * u + math.sin(angle) * v
        d = p + transverse_scale * s
        xhat = -p + transverse_scale * s
        d /= vector_norm(d)
        xhat /= vector_norm(xhat)
        e1, e2 = orthonormal_basis_perp(d)
        geometries.append((d, xhat, np.column_stack([e1, e2])))
    return geometries
