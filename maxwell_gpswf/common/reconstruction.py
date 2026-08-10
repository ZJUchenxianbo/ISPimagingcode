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


def select_complete_gpswf_multiplets(
    modes: list[Mode],
    chi_by_pair: dict[tuple[int, int], float],
    max_modes: int,
    *,
    eligible: np.ndarray | None = None,
    alpha_plateau_rtol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Select a spectral prefix without splitting an ``(ell, n)`` multiplet.

    ``|alpha|`` remains the primary ordering criterion.  Values that differ by
    no more than ``alpha_plateau_rtol`` are treated as one numerical plateau
    and ordered by the Sturm-Liouville eigenvalue ``chi``.  The requested
    ``max_modes`` is an upper bound: selection stops before the next complete
    multiplet would exceed it.
    """
    if max_modes < 0:
        raise ValueError("max_modes must be nonnegative")
    if alpha_plateau_rtol < 0.0:
        raise ValueError("alpha_plateau_rtol must be nonnegative")

    n_total = len(modes)
    if eligible is None:
        eligible_mask = np.ones(n_total, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)
        if eligible_mask.shape != (n_total,):
            raise ValueError("eligible must have shape (len(modes),)")

    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for index, mode in enumerate(modes):
        grouped_indices.setdefault((mode.ell, mode.n), []).append(index)

    groups: list[dict[str, Any]] = []
    for pair, indices in grouped_indices.items():
        ell, n = pair
        expected_m = list(range(-ell, ell + 1))
        actual_m = sorted(modes[index].m for index in indices)
        if actual_m != expected_m:
            raise ValueError(
                f"incomplete GPSWF multiplet {pair}: expected m={expected_m}, "
                f"got m={actual_m}"
            )
        pair_eligibility = eligible_mask[indices]
        if np.any(pair_eligibility) and not np.all(pair_eligibility):
            raise ValueError(f"eligible splits GPSWF multiplet {pair}")
        if not np.all(pair_eligibility):
            continue
        if pair not in chi_by_pair:
            raise ValueError(f"missing chi value for GPSWF multiplet {pair}")
        alpha_values = np.asarray([abs(modes[index].alpha) for index in indices])
        if not np.allclose(alpha_values, alpha_values[0], rtol=1e-12, atol=1e-14):
            raise ValueError(f"alpha is inconsistent inside GPSWF multiplet {pair}")
        groups.append({
            "pair": pair,
            "indices": indices,
            "size": len(indices),
            "alpha_abs": float(alpha_values[0]),
            "chi": float(chi_by_pair[pair]),
        })

    groups.sort(key=lambda group: -float(group["alpha_abs"]))
    ordered_groups: list[dict[str, Any]] = []
    plateau: list[dict[str, Any]] = []
    plateau_alpha = 0.0

    def flush_plateau() -> None:
        if plateau:
            ordered_groups.extend(sorted(
                plateau,
                key=lambda group: (
                    float(group["chi"]),
                    int(group["pair"][0]),
                    int(group["pair"][1]),
                ),
            ))

    for group in groups:
        alpha_abs = float(group["alpha_abs"])
        if not plateau:
            plateau = [group]
            plateau_alpha = alpha_abs
            continue
        scale = max(abs(plateau_alpha), abs(alpha_abs), 1e-300)
        if abs(alpha_abs - plateau_alpha) <= alpha_plateau_rtol * scale:
            plateau.append(group)
        else:
            flush_plateau()
            plateau = [group]
            plateau_alpha = alpha_abs
    flush_plateau()

    retained = np.zeros(n_total, dtype=bool)
    retained_count = 0
    retained_multiplets = 0
    for group in ordered_groups:
        group_size = int(group["size"])
        if retained_count + group_size > max_modes:
            break
        retained[np.asarray(group["indices"], dtype=int)] = True
        retained_count += group_size
        retained_multiplets += 1

    metadata: dict[str, int | float] = {
        "requested_max_modes": int(max_modes),
        "retained_modes": int(retained_count),
        "retained_multiplets": int(retained_multiplets),
        "partial_multiplets": 0,
        "eligible_multiplets": int(len(groups)),
        "alpha_plateau_rtol": float(alpha_plateau_rtol),
    }
    return retained, metadata


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
