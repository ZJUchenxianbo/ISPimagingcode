#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime diagnostics for Maxwell GPSWF reconstructions.

These helpers are intended to be called inside figure scripts, using the exact
intermediate arrays that produced the plotted image.  They replace the older
paper-table diagnostics, which were useful for reproducing formulas but poor at
locating failures in a specific reconstruction.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.utils import make_table, vector_norm


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def node_diagnostics(mock_distances: np.ndarray | None) -> dict[str, float]:
    """Summarise mock-quadrature mismatch distances."""
    if mock_distances is None:
        return {
            "mock_distance_mean": math.nan,
            "mock_distance_max": math.nan,
            "mock_distance_p95": math.nan,
        }
    distances = np.asarray(mock_distances, dtype=float)
    if distances.size == 0:
        return {
            "mock_distance_mean": math.nan,
            "mock_distance_max": math.nan,
            "mock_distance_p95": math.nan,
        }
    return {
        "mock_distance_mean": float(np.mean(distances)),
        "mock_distance_max": float(np.max(distances)),
        "mock_distance_p95": float(np.percentile(distances, 95)),
    }


def mode_diagnostics(
    modes: list[Any],
    retained: np.ndarray,
    *,
    target_nodes_count: int,
) -> dict[str, float | int]:
    """Summarise retained GPSWF modes and alpha values."""
    retained = np.asarray(retained, dtype=bool)
    alpha_abs = np.asarray([abs(mode.alpha) for mode in modes], dtype=float)
    total_modes = int(len(modes))
    retained_modes = int(np.sum(retained))
    retained_alpha = alpha_abs[retained]
    retained_ells = np.asarray([mode.ell for mode, keep in zip(modes, retained) if keep], dtype=int)
    retained_ns = np.asarray([mode.n for mode, keep in zip(modes, retained) if keep], dtype=int)
    alpha_max = float(np.max(alpha_abs)) if alpha_abs.size else math.nan
    alpha_min_retained = float(np.min(retained_alpha)) if retained_alpha.size else math.nan
    alpha_median_retained = float(np.median(retained_alpha)) if retained_alpha.size else math.nan
    return {
        "total_modes": total_modes,
        "retained_modes": retained_modes,
        "retained_ratio": retained_modes / max(total_modes, 1),
        "retained_ell_max": int(np.max(retained_ells)) if retained_ells.size else -1,
        "retained_n_max": int(np.max(retained_ns)) if retained_ns.size else -1,
        "alpha_max": alpha_max,
        "alpha_min_retained": alpha_min_retained,
        "alpha_median_retained": alpha_median_retained,
        "target_nodes_per_total_modes": float(target_nodes_count) / max(total_modes, 1),
        "target_nodes_per_retained_modes": float(target_nodes_count) / max(retained_modes, 1),
    }


def projection_diagnostics(
    basis_matrix: np.ndarray,
    target_weights: np.ndarray,
    retained: np.ndarray,
    *,
    max_gram_modes: int = 1200,
) -> dict[str, float | str]:
    """Diagnose the retained discrete projection Gram matrix.

    Computing ``A^H W A`` is quadratic in the retained mode count.  Large cases
    are marked as skipped so diagnostics do not become more expensive than the
    experiment being diagnosed.
    """
    retained = np.asarray(retained, dtype=bool)
    n_retained = int(np.sum(retained))
    if n_retained <= 0:
        return {
            "gram_offdiag_ratio": math.nan,
            "gram_cond": math.nan,
            "projection_branch": "empty",
            "regularization": math.nan,
        }
    if n_retained > int(max_gram_modes):
        return {
            "gram_offdiag_ratio": math.nan,
            "gram_cond": math.nan,
            "projection_branch": "skipped_large",
            "regularization": math.nan,
        }

    A = np.asarray(basis_matrix[:, retained], dtype=np.complex128)
    weights = np.asarray(target_weights, dtype=float)
    AWA = np.conj(A).T @ (weights[:, None] * A)
    diag_AWA = np.diag(np.diag(AWA))
    diag_norm = max(vector_norm(diag_AWA), 1e-14)
    gram_offdiag_ratio = vector_norm(AWA - diag_AWA) / diag_norm
    try:
        gram_cond = float(np.linalg.cond(AWA))
    except np.linalg.LinAlgError:
        gram_cond = math.inf
    regularization = 1e-10 * abs(complex(np.trace(AWA))) / max(n_retained, 1)
    projection_branch = "diagonal" if gram_offdiag_ratio < 0.01 else "solve"
    return {
        "gram_offdiag_ratio": float(gram_offdiag_ratio),
        "gram_cond": gram_cond,
        "projection_branch": projection_branch,
        "regularization": float(regularization),
    }


def data_diagnostics(component_data: np.ndarray) -> dict[str, float]:
    """Summarise scalar Fourier data used by one reconstruction."""
    data = np.asarray(component_data, dtype=np.complex128)
    return {
        "data_norm": vector_norm(data),
        "data_max_abs": float(np.max(np.abs(data))) if data.size else math.nan,
    }


def coefficient_diagnostics(coeffs: np.ndarray) -> dict[str, float]:
    """Summarise recovered modal coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    return {
        "coeff_norm": vector_norm(coeffs),
        "coeff_max_abs": float(np.max(np.abs(coeffs))) if coeffs.size else math.nan,
    }


def image_diagnostics(
    image: np.ndarray,
    truth: np.ndarray,
    disk_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarise reconstruction intensity and target/background leakage."""
    img = np.asarray(image)
    truth_arr = np.asarray(truth)
    if disk_mask is None:
        valid = np.ones(img.shape, dtype=bool)
    else:
        valid = np.asarray(disk_mask, dtype=bool)
    target_mask = valid & (np.abs(truth_arr) > 1e-12)
    background_mask = valid & ~target_mask
    abs_img = np.abs(img)

    def _mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_img[mask])) if np.any(mask) else math.nan

    def _p95(mask: np.ndarray) -> float:
        return float(np.percentile(abs_img[mask], 95)) if np.any(mask) else math.nan

    return {
        "image_min": float(np.min(np.real(img))) if img.size else math.nan,
        "image_max": float(np.max(np.real(img))) if img.size else math.nan,
        "image_max_abs": float(np.max(abs_img)) if img.size else math.nan,
        "image_l2": vector_norm(img),
        "background_mean_abs": _mean(background_mask),
        "background_p95_abs": _p95(background_mask),
        "target_mean_abs": _mean(target_mask),
        "target_p95_abs": _p95(target_mask),
    }


def collect_reconstruction_diagnostics(
    *,
    case: dict[str, Any],
    modes: list[Any],
    retained: np.ndarray,
    target_nodes: np.ndarray,
    p_nodes: np.ndarray,
    mock_distances: np.ndarray | None,
    basis_matrix: np.ndarray,
    target_weights: np.ndarray,
    component_data: np.ndarray,
    coeffs: np.ndarray,
    image: np.ndarray,
    truth: np.ndarray,
    disk_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Collect one CSV-ready diagnostic row for a reconstruction."""
    row: dict[str, Any] = dict(case)
    row["target_nodes"] = int(np.asarray(target_nodes).shape[0])
    row["p_nodes"] = int(np.asarray(p_nodes).shape[0])
    row.update(node_diagnostics(mock_distances))
    row.update(mode_diagnostics(modes, retained, target_nodes_count=row["target_nodes"]))
    row.update(projection_diagnostics(basis_matrix, target_weights, retained))
    row.update(data_diagnostics(component_data))
    row.update(coefficient_diagnostics(coeffs))
    row.update(image_diagnostics(image, truth, disk_mask))
    return row


def write_diagnostics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write scalar diagnostics to CSV using the project table fallback."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    make_table(rows).to_csv(path, index=False)


def save_diagnostics_npz(rows: list[dict[str, Any]], path: Path) -> None:
    """Save scalar diagnostic rows as a lightweight NPZ mirror."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row.get(key, "") for row in rows]
        if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            arrays[key] = np.asarray([_finite_float(value) for value in values], dtype=float)
        else:
            arrays[key] = np.asarray([str(value) for value in values], dtype=object)
    np.savez_compressed(path, **arrays)


def plot_diagnostic_curves(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    """Plot compact scalar diagnostics for quick failure localisation."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))

    def values(key: str) -> np.ndarray:
        return np.asarray([_finite_float(row.get(key)) for row in rows], dtype=float)

    labels = [str(row.get("case_id", idx)) for idx, row in enumerate(rows)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    series = [
        ("retained_modes", "Retained modes"),
        ("coeff_norm", "Coefficient norm"),
        ("background_p95_abs", "Background p95 abs"),
        ("image_max_abs", "Image max abs"),
    ]
    for ax, (key, ylabel) in zip(axes.ravel(), series):
        ax.plot(x, values(key), marker="o", linewidth=1.2)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)
