#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared GPSWF reconstruction utilities.

Pulled from duplicated code across figure scripts (5/6/7).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.gpswf import collect_alpha_pairs_cached, solve_ball_gpswf, modal_matrix
from common.phantom import Mode


def build_gpswf_modes(
    C: float,
    K: int,
    ell_max: int,
    n_modes_per_ell: int,
    quad_order: int,
    r_eval_count: int,
    cache_dir: Path,
) -> list[Mode]:
    """Build the full list of GPSWF Mode objects for given parameters.

    Returns ``modes`` of length ``n_modes_per_ell * (ell_max+1)²``, each
    with populated ``alpha`` and ``beta``.
    """
    alpha_df = collect_alpha_pairs_cached(
        C, K, ell_max, n_modes_per_ell, quad_order=quad_order,
        r_eval_count=r_eval_count, cache_dir=cache_dir,
    )
    alpha_lookup = {
        (int(row["ell"]), int(row["n"])): complex(float(row["alpha_real"]), float(row["alpha_imag"]))
        for _, row in alpha_df.iterrows()
    }
    modes: list[Mode] = []
    for ell in range(ell_max + 1):
        _, beta = solve_ball_gpswf(C, ell, K, n_modes=n_modes_per_ell)
        for n in range(beta.shape[1]):
            a = alpha_lookup[(ell, n)]
            for m in range(-ell, ell + 1):
                modes.append(Mode(ell=ell, n=n, m=m, alpha=a, beta=beta[:, n]))
    return modes


def truncate_modes(
    modes: list[Mode],
    epsilon: float,
    N_cap: int,
) -> np.ndarray:
    """Three-layer truncation: alpha_abs > epsilon*max → N_cap.

    Returns a boolean ``retained`` array of length ``len(modes)``.
    """
    alpha_abs = np.asarray([abs(m.alpha) for m in modes], dtype=float)
    retained = alpha_abs > epsilon * float(np.max(alpha_abs))
    if np.sum(retained) > N_cap:
        order = np.argsort(-alpha_abs)
        keep = order[:N_cap]
        retained = np.zeros(len(modes), dtype=bool)
        retained[keep] = True
    return retained


def make_xy_grid(grid_size: int) -> np.ndarray:
    """Return (grid_size², 3) array of (x, y, z=0) points on [-1,1]²."""
    xs = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])


def gpswf_reconstruct_image(
    component_data: np.ndarray,
    target_basis: np.ndarray,
    target_weights: np.ndarray,
    modes: list[Mode],
    retained: np.ndarray,
    image_matrix: np.ndarray,
    grid_size: int,
    disk_mask: np.ndarray | None = None,
) -> np.ndarray:
    """GPSWF quadrature projection → 2D image.

    Returns ``(grid_size, grid_size)`` complex array.
    """
    from common.gpswf import quadrature_modal_coefficients

    coeffs = quadrature_modal_coefficients(
        component_data, target_basis, target_weights, modes, retained)
    rec = (image_matrix @ coeffs).reshape(grid_size, grid_size)
    if disk_mask is not None:
        rec[~disk_mask] = 0.0
    return rec
