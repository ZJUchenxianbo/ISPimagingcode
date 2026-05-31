#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared direct-sampling utilities for 2D inverse scattering experiments.

The routines here keep the imaging part independent from the forward solver:
obstacle BEM data and point-scatterer data can be passed through the same
backpropagation/orthogonality-sampling indicator.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from scattering_common import PI2, Array, CArray, direction_vectors


def aperture_measure(alpha: float) -> float:
    """Return the angular length of an aperture with half-width alpha."""
    if not (0.0 < alpha <= math.pi):
        raise ValueError("alpha must be in (0, pi]")
    return PI2 if math.isclose(alpha, math.pi, rel_tol=0.0, abs_tol=1e-14) else 2.0 * alpha


def aperture_angles(center: float, alpha: float, n_obs: int) -> Array:
    """Observation angles for a full or finite aperture.

    Full aperture uses endpoint=False on [0, 2*pi). Finite aperture includes
    both endpoints so trapezoidal quadrature can be used.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    length = aperture_measure(alpha)
    if math.isclose(length, PI2, rel_tol=0.0, abs_tol=1e-14):
        return np.linspace(0.0, PI2, n_obs, endpoint=False)
    return center + np.linspace(-alpha, alpha, n_obs)


def observation_weights(n_obs: int, aperture_length: float) -> Array:
    """Quadrature weights for uniformly sampled observation directions.

    The full aperture grid is periodic and endpoint-free, so the rectangle rule
    gives weight 2*pi/n. Finite apertures include both endpoints and use the
    trapezoidal rule, which avoids over-weighting the two edge directions.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if aperture_length <= 0.0:
        raise ValueError("aperture_length must be positive")
    if math.isclose(aperture_length, PI2, rel_tol=0.0, abs_tol=1e-14):
        return np.full(n_obs, aperture_length / n_obs, dtype=float)

    h = aperture_length / (n_obs - 1)
    weights = np.full(n_obs, h, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def normalize_indicator(indicator: Array) -> Array:
    """Normalize a real indicator image to max value 1."""
    max_value = float(np.max(indicator))
    if max_value <= 1e-14:
        return indicator.copy()
    return indicator / max_value


def image_extent(x_grid: Array, y_grid: Array) -> list[float]:
    """Return imshow extent from 1D x/y grids."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    return [float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])]


def plot_indicator_image(
    ax,
    image: Array,
    x_grid: Array | None = None,
    y_grid: Array | None = None,
    *,
    extent: list[float] | tuple[float, float, float, float] | None = None,
    title: str | None = None,
    cmap: str = "jet",
    vmin: float = 0.0,
    vmax: float = 1.0,
    interpolation: str = "bilinear",
    axis_off: bool = True,
):
    """Plot a normalized imaging indicator with the project's default style."""
    if extent is None:
        if x_grid is None or y_grid is None:
            raise ValueError("either extent or both x_grid and y_grid must be provided")
        extent = image_extent(x_grid, y_grid)
    im = ax.imshow(
        image,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if axis_off:
        ax.set_axis_off()
    return im


def direct_sampling_indicator(
    farfield_matrix: CArray,
    k: float,
    obs_angles: Array,
    incident_angles: Array,
    x_grid: Array,
    y_grid: Array,
    aperture_length: float = PI2,
    power: float = 1.0,
    block_size: int = 32768,
    phase_sign: float = 1.0,
    normalize: bool = True,
) -> Array:
    """Compute a multi-incident direct/orthogonality-sampling indicator.

    The discrete indicator is

        sum_d | sum_xhat w_xhat u_inf(xhat,d) exp(i*k*xhat.y) |**power.

    Computation is chunked over imaging points, so larger grids do not require
    storing the full observation-by-grid phase matrix at once.
    """
    return direct_sampling_indicators(
        [farfield_matrix],
        k,
        obs_angles,
        incident_angles,
        x_grid,
        y_grid,
        aperture_length,
        power,
        block_size,
        phase_sign,
        normalize,
    )[0]


def direct_sampling_indicators(
    farfield_matrices: Iterable[CArray],
    k: float,
    obs_angles: Array,
    incident_angles: Array,
    x_grid: Array,
    y_grid: Array,
    aperture_length: float = PI2,
    power: float = 1.0,
    block_size: int = 32768,
    phase_sign: float = 1.0,
    normalize: bool = True,
) -> list[Array]:
    """Compute several indicators while reusing the same observation phases."""
    farfields = [np.asarray(item, dtype=np.complex128) for item in farfield_matrices]
    if not farfields:
        raise ValueError("farfield_matrices must contain at least one matrix")
    obs = np.asarray(obs_angles, dtype=float)
    inc = np.asarray(incident_angles, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    expected_shape = (obs.size, inc.size)
    for farfield in farfields:
        if farfield.shape != expected_shape:
            raise ValueError(
                "each farfield matrix shape must be (len(obs_angles), len(incident_angles)); "
                f"got {farfield.shape}, expected {expected_shape}"
            )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    xhat = direction_vectors(obs)
    obs_w = observation_weights(obs.size, float(aperture_length))
    inc_weight = PI2 / max(inc.size, 1)

    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    values_list = [np.empty(pts.shape[0], dtype=float) for _ in farfields]

    weighted_farfield_ts = [farfield.T * obs_w[None, :] for farfield in farfields]
    sign = 1.0 if phase_sign >= 0.0 else -1.0
    for start in range(0, pts.shape[0], block_size):
        stop = min(start + block_size, pts.shape[0])
        phase = np.exp(1j * sign * k * (xhat @ pts[start:stop].T))
        for values, weighted_farfield_t in zip(values_list, weighted_farfield_ts):
            reduced = weighted_farfield_t @ phase
            values[start:stop] = inc_weight * np.sum(np.abs(reduced) ** float(power), axis=0)

    images = [values.reshape(X.shape) for values in values_list]
    return [normalize_indicator(image) for image in images] if normalize else images



