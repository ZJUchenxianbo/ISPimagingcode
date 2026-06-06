#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified forward solvers for 2D acoustic scattering experiments.

The public far-field convention is always
``(len(obs_angles), len(incident_angles))``.  The module groups the reusable
forward models in one place:

* sound-soft obstacles by boundary integral equations,
* penetrable media by Born or Lippmann-Schwinger volume integral equations,
* point scatterers by independent amplitudes or a Foldy-Lax interaction model.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.special import hankel1

from common.scattering import PI2, Array, CArray, direction_vectors
from common.targets import BoundaryGeometry, params_to_geometry

ObstacleMethod = Literal["single_layer", "double_layer", "combined_field"]
MediumMethod = Literal["born", "lippmann_schwinger"]
PointMethod = Literal["independent", "foldy_lax"]


def _validate_k_and_angles(k: float, incident_angles: Array, obs_angles: Array) -> tuple[Array, Array]:
    if k <= 0.0:
        raise ValueError("k must be positive")
    inc = np.asarray(incident_angles, dtype=float)
    obs = np.asarray(obs_angles, dtype=float)
    if inc.ndim != 1:
        raise ValueError("incident_angles must be one-dimensional")
    if obs.ndim != 1:
        raise ValueError("obs_angles must be one-dimensional")
    return inc, obs


def _green_2d(k: float, distances: Array) -> CArray:
    return 0.25j * hankel1(0, k * distances)


def _farfield_constant(k: float) -> complex:
    return np.exp(1j * np.pi / 4.0) / np.sqrt(8.0 * np.pi * k)


# ---------------------------------------------------------------------------
# 障碍物 BEM 底层算子 (单层势)
# ---------------------------------------------------------------------------

def plane_wave(x: Array, k: float, d: Array) -> CArray:
    """Incident plane wave exp(i*k*d.x) sampled at x."""
    return np.exp(1j * k * (x @ d))


def plane_wave_matrix(x: Array, k: float, incident_angles: Array) -> CArray:
    """Incident plane waves for all incident directions sampled at x.

    The returned matrix has shape ``(n_points, n_incident_angles)``.
    """
    inc = np.asarray(incident_angles, dtype=float)
    if inc.ndim != 1:
        raise ValueError("incident_angles must be one-dimensional")
    directions = direction_vectors(inc)
    return np.exp(1j * k * (np.asarray(x, dtype=float) @ directions.T))


def _diag_single_layer_integral(k: float, h: float) -> complex:
    """Approximate the diagonal single-layer integral over one collocation cell."""
    if h <= 0.0:
        return 0.0 + 0.0j

    def f_re(s: float) -> float:
        return float(np.real(0.25j * hankel1(0, k * s)))

    def f_im(s: float) -> float:
        return float(np.imag(0.25j * hankel1(0, k * s)))

    a, b = 0.0, 0.5 * h
    re_val = quad(f_re, a, b, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    im_val = quad(f_im, a, b, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    return 2.0 * (re_val + 1j * im_val)


def build_single_layer_matrix(geom: BoundaryGeometry, k: float) -> CArray:
    """Build the single-layer boundary integral matrix for sound-soft obstacles."""
    n = geom.x.shape[0]
    A = np.empty((n, n), dtype=complex)
    for i in range(n):
        diff = geom.x[i][None, :] - geom.x
        rho = np.linalg.norm(diff, axis=1)
        row = 0.25j * hankel1(0, k * rho) * geom.ds
        row[i] = _diag_single_layer_integral(k, float(geom.ds[i]))
        A[i, :] = row
    return A


def single_layer_farfield_operator(geom: BoundaryGeometry, k: float, obs_angles: Array) -> CArray:
    """Build the linear map from boundary density to far-field pattern."""
    xhat = direction_vectors(np.asarray(obs_angles, dtype=float))
    const = _farfield_constant(k)
    phase = np.exp(-1j * k * (xhat @ geom.x.T))
    return const * phase * geom.ds[None, :]


# ---------------------------------------------------------------------------
# 障碍物 BEM 底层算子 (双层势)
# ---------------------------------------------------------------------------


def build_double_layer_matrix(geom: BoundaryGeometry, k: float) -> CArray:
    """Build the principal-value double-layer matrix for sound-soft obstacles.

    Off-diagonal entries use
    ``d Phi(x_i, y_j) / d nu(y_j)``.  The jump term is not included here; it is
    added explicitly as ``0.5 * I`` in the exterior Dirichlet equation.
    """
    n = geom.x.shape[0]
    K = np.zeros((n, n), dtype=complex)
    for i in range(n):
        diff = geom.x[i][None, :] - geom.x
        rho = np.linalg.norm(diff, axis=1)
        mask = rho > 0.0
        normal_projection = np.sum(diff[mask] * geom.normal[mask], axis=1) / rho[mask]
        K[i, mask] = 0.25j * k * hankel1(1, k * rho[mask]) * normal_projection * geom.ds[mask]
    return K


def double_layer_farfield_operator(geom: BoundaryGeometry, k: float, obs_angles: Array) -> CArray:
    """Build the far-field map for a double-layer density."""
    xhat = direction_vectors(np.asarray(obs_angles, dtype=float))
    phase = np.exp(-1j * k * (xhat @ geom.x.T))
    normal_projection = xhat @ geom.normal.T
    return _farfield_constant(k) * (-1j * k * normal_projection) * phase * geom.ds[None, :]


def solve_obstacle_farfield(
    params: Array,
    k: float,
    n_per_obstacle: int,
    incident_angles: Array,
    obs_angles: Array,
    *,
    n_obstacles: int = 3,
    method: ObstacleMethod = "single_layer",
    eta: float | None = None,
    stabilization: float = 1e-12,
) -> CArray:
    """Compute far-field data for sound-soft obstacles.

    ``single_layer`` solves ``S sigma = -u_i``.  ``double_layer`` solves
    ``(0.5 I + K) phi = -u_i``.  ``combined_field`` uses
    ``(0.5 I + K - i eta S) phi = -u_i`` and is usually more robust away from
    the simple tests; by default ``eta = k``.
    """
    inc, obs = _validate_k_and_angles(k, incident_angles, obs_angles)
    if method not in ("single_layer", "double_layer", "combined_field"):
        raise ValueError("method must be 'single_layer', 'double_layer', or 'combined_field'")

    geom = params_to_geometry(params, n_per_obstacle, n_obstacles=n_obstacles)
    rhs = -plane_wave_matrix(geom.x, k, inc)
    ident = np.eye(geom.x.shape[0], dtype=complex)

    S: CArray | None = None
    Sinf: CArray | None = None
    if method in ("single_layer", "combined_field"):
        S = build_single_layer_matrix(geom, k)
        Sinf = single_layer_farfield_operator(geom, k, obs)

    if method == "single_layer":
        system = S
        output = Sinf
    else:
        K = build_double_layer_matrix(geom, k)
        Dinf = double_layer_farfield_operator(geom, k, obs)
        if method == "double_layer":
            system = 0.5 * ident + K
            output = Dinf
        else:
            eta_val = float(k if eta is None else eta)
            system = 0.5 * ident + K - 1j * eta_val * S
            output = Dinf - 1j * eta_val * Sinf

    if system is None or output is None:
        raise RuntimeError("internal obstacle solver setup failed")
    stab = float(stabilization) * max(float(np.linalg.norm(system, ord=2)), 1.0)
    density = solve(system + stab * ident, rhs, assume_a="gen")
    return output @ density


def solve_forward_farfield(
    params: Array,
    k: float,
    n_per_obstacle: int,
    incident_angles: Array,
    obs_angles: Array,
    n_obstacles: int = 3,
) -> CArray:
    """Compute the far-field matrix for sound-soft obstacles (single-layer).

    Convenience wrapper around ``solve_obstacle_farfield`` with the historical
    positional-argument API used by most experiment scripts.
    """
    return solve_obstacle_farfield(
        params, k, n_per_obstacle, incident_angles, obs_angles,
        n_obstacles=n_obstacles, method="single_layer",
    )


def solve_point_scatterer_farfield(
    points: Array,
    strengths: CArray,
    k: float,
    incident_angles: Array,
    obs_angles: Array,
    *,
    method: PointMethod = "independent",
) -> CArray:
    """Compute far-field data for isotropic point scatterers.

    ``independent`` uses
    ``sum_j q_j exp(-i k xhat.z_j) exp(i k d.z_j)``.  ``foldy_lax`` keeps the
    same far-field amplitude convention but solves the point interaction system
    before evaluating the outgoing sum.
    """
    inc, obs = _validate_k_and_angles(k, incident_angles, obs_angles)
    points = np.asarray(points, dtype=float)
    strengths = np.asarray(strengths, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if strengths.shape != (points.shape[0],):
        raise ValueError("strengths must have shape (n_points,)")
    if method not in ("independent", "foldy_lax"):
        raise ValueError("method must be 'independent' or 'foldy_lax'")

    xhat = direction_vectors(obs)
    dhat = direction_vectors(inc)
    receive_phase = np.exp(-1j * k * (xhat @ points.T))
    incident_field = np.exp(1j * k * (dhat @ points.T)).T

    if method == "independent" or points.shape[0] == 1:
        point_fields = incident_field
    else:
        diff = points[:, None, :] - points[None, :, :]
        rho = np.linalg.norm(diff, axis=2)
        interaction = np.zeros_like(rho, dtype=complex)
        mask = rho > 0.0
        interaction[mask] = _green_2d(k, rho[mask])
        interaction *= strengths[None, :]
        point_fields = solve(np.eye(points.shape[0], dtype=complex) - interaction, incident_field, assume_a="gen")

    return receive_phase @ (strengths[:, None] * point_fields)


def disk_contrast(
    x_grid: Array,
    y_grid: Array,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 0.2,
    contrast: float = 1.0,
    profile: Literal["homogeneous", "inhomogeneous"] = "homogeneous",
    modulation_amplitude: float = 0.0,
    mode_x: float = 1.0,
    mode_y: float = 1.0,
) -> CArray:
    """Build a homogeneous or smoothly modulated disk contrast on a tensor grid."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    dx = X - float(center[0])
    dy = Y - float(center[1])
    mask = dx * dx + dy * dy <= float(radius) ** 2
    q = np.zeros_like(X, dtype=np.complex128)
    if profile == "homogeneous":
        q[mask] = complex(contrast)
    elif profile == "inhomogeneous":
        modulation = 1.0 + float(modulation_amplitude) * (
            np.cos(float(mode_x) * np.pi * dx / float(radius))
            * np.sin(float(mode_y) * np.pi * dy / float(radius))
        )
        q[mask] = complex(contrast) * modulation[mask]
    else:
        raise ValueError("profile must be 'homogeneous' or 'inhomogeneous'")
    return q


def _uniform_cell_area(x_grid: Array, y_grid: Array) -> float:
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional")
    if x_grid.size < 2 or y_grid.size < 2:
        raise ValueError("x_grid and y_grid must have at least two points")
    hx = np.diff(x_grid)
    hy = np.diff(y_grid)
    if not np.allclose(hx, hx[0]) or not np.allclose(hy, hy[0]):
        raise ValueError("medium solver currently expects uniform x_grid and y_grid")
    return abs(float(hx[0] * hy[0]))


def _diag_volume_green_integral(k: float, cell_area: float) -> complex:
    """Approximate int_cell Phi(0,y) dy by an equal-area disk integral."""
    radius = np.sqrt(float(cell_area) / np.pi)

    def f_re(r: float) -> float:
        return float(np.real(_green_2d(k, np.asarray(r))) * 2.0 * np.pi * r)

    def f_im(r: float) -> float:
        return float(np.imag(_green_2d(k, np.asarray(r))) * 2.0 * np.pi * r)

    re_val = quad(f_re, 0.0, radius, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    im_val = quad(f_im, 0.0, radius, points=[0.0], limit=200, epsabs=1e-10, epsrel=1e-10)[0]
    return re_val + 1j * im_val


def solve_medium_farfield(
    contrast: CArray,
    x_grid: Array,
    y_grid: Array,
    k: float,
    incident_angles: Array,
    obs_angles: Array,
    *,
    method: MediumMethod = "born",
    active_tol: float = 0.0,
    max_unknowns: int = 2500,
) -> CArray:
    """Compute far-field data for a penetrable medium contrast q(x).

    The model is ``Delta u + k^2 (1 + q) u = 0``.  ``born`` uses
    ``u ~= u_i`` in the volume integral.  ``lippmann_schwinger`` solves
    ``u = u_i + k^2 G(q u)`` on the active grid cells.
    """
    inc, obs = _validate_k_and_angles(k, incident_angles, obs_angles)
    if method not in ("born", "lippmann_schwinger"):
        raise ValueError("method must be 'born' or 'lippmann_schwinger'")

    contrast = np.asarray(contrast, dtype=np.complex128)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if contrast.shape != (y_grid.size, x_grid.size):
        raise ValueError("contrast shape must be (len(y_grid), len(x_grid))")

    cell_area = _uniform_cell_area(x_grid, y_grid)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    active = np.abs(contrast) > float(active_tol)
    if not np.any(active):
        return np.zeros((obs.size, inc.size), dtype=complex)

    points = np.column_stack([X[active], Y[active]])
    q = contrast[active]
    if method == "lippmann_schwinger" and points.shape[0] > int(max_unknowns):
        raise ValueError(
            f"lippmann_schwinger has {points.shape[0]} active cells; "
            f"increase max_unknowns if this dense solve is intentional"
        )

    dhat = direction_vectors(inc)
    incident_field = np.exp(1j * k * (points @ dhat.T))
    if method == "born":
        total_field = incident_field
    else:
        diff = points[:, None, :] - points[None, :, :]
        rho = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(rho, 1.0)
        G = _green_2d(k, rho) * cell_area
        np.fill_diagonal(G, _diag_volume_green_integral(k, cell_area))
        system = np.eye(points.shape[0], dtype=complex) - (k * k) * G * q[None, :]
        total_field = solve(system, incident_field, assume_a="gen")

    xhat = direction_vectors(obs)
    phase = np.exp(-1j * k * (xhat @ points.T))
    source = q[:, None] * total_field * cell_area
    return (k * k) * _farfield_constant(k) * (phase @ source)
