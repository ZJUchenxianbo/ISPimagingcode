#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct sampling indicators from recovered Fourier data."""
from __future__ import annotations

from typing import Any

import numpy as np

from common.utils import vector_norm


def direct_sampling_component_indicator(
    component_data: np.ndarray,
    p_nodes: np.ndarray,
    weights: np.ndarray,
    points: np.ndarray,
    C: float,
    *,
    block_size: int = 2048,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a DSM indicator from one recovered tensor Fourier component.

    The indicator is the weighted phase backprojection

        I(z) = |sum_i w_i Qhat(p_i) exp(i C p_i.z)|.

    It uses the same recovered Fourier data as the modal reconstructions.  When
    ``normalize`` is True the returned image is divided by its maximum value, so
    it should be interpreted as a location indicator rather than a contrast
    reconstruction.
    """
    data = np.asarray(component_data, dtype=np.complex128)
    nodes = np.asarray(p_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    points = np.asarray(points, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if data.shape != (nodes.shape[0],):
        raise ValueError("component_data must have shape (n_nodes,)")
    if weights.shape != (nodes.shape[0],):
        raise ValueError("weights must have shape (n_nodes,)")

    weighted_data = weights * data
    backprojection = np.empty(points.shape[0], dtype=np.complex128)
    for start in range(0, points.shape[0], int(block_size)):
        stop = min(start + int(block_size), points.shape[0])
        phase = np.exp(1j * float(C) * (points[start:stop] @ nodes.T))
        backprojection[start:stop] = phase @ weighted_data

    indicator = np.abs(backprojection)
    raw_max = float(np.max(indicator)) if indicator.size else 0.0
    if normalize and raw_max > 0.0:
        image = indicator / raw_max
    else:
        image = indicator
    meta = {
        "dsm_nodes": int(nodes.shape[0]),
        "dsm_normalized": int(bool(normalize)),
        "dsm_raw_max": raw_max,
        "dsm_raw_l2": vector_norm(indicator),
        "data_norm": vector_norm(data),
        "data_max_abs": float(np.max(np.abs(data))) if data.size else float("nan"),
        "coeff_norm": vector_norm(weighted_data),
        "coeff_max_abs": float(np.max(np.abs(weighted_data))) if weighted_data.size else float("nan"),
    }
    return image.astype(np.complex128), meta


def direct_sampling_tensor_indicator(
    tensor_data: np.ndarray,
    p_nodes: np.ndarray,
    weights: np.ndarray,
    points: np.ndarray,
    C: float,
    *,
    block_size: int = 2048,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a DSM indicator from all recovered tensor Fourier components."""
    data = np.asarray(tensor_data, dtype=np.complex128)
    nodes = np.asarray(p_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    points = np.asarray(points, dtype=float)
    if data.ndim != 2:
        raise ValueError("tensor_data must have shape (n_nodes, n_components)")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if data.shape[0] != nodes.shape[0]:
        raise ValueError("tensor_data and p_nodes must have the same first dimension")
    if weights.shape != (nodes.shape[0],):
        raise ValueError("weights must have shape (n_nodes,)")

    weighted_data = weights[:, None] * data
    indicator = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], int(block_size)):
        stop = min(start + int(block_size), points.shape[0])
        phase = np.exp(1j * float(C) * (points[start:stop] @ nodes.T))
        backprojection = phase @ weighted_data
        indicator[start:stop] = np.linalg.norm(backprojection, axis=1)

    raw_max = float(np.max(indicator)) if indicator.size else 0.0
    if normalize and raw_max > 0.0:
        image = indicator / raw_max
    else:
        image = indicator
    meta = {
        "dsm_nodes": int(nodes.shape[0]),
        "dsm_components": int(data.shape[1]),
        "dsm_normalized": int(bool(normalize)),
        "dsm_raw_max": raw_max,
        "dsm_raw_l2": vector_norm(indicator),
        "data_norm": vector_norm(data),
        "data_max_abs": float(np.max(np.abs(data))) if data.size else float("nan"),
        "coeff_norm": vector_norm(weighted_data),
        "coeff_max_abs": float(np.max(np.abs(weighted_data))) if weighted_data.size else float("nan"),
    }
    return image.astype(np.complex128), meta


def direct_sampling_farfield_indicator(
    farfield_data: np.ndarray,
    p_nodes: np.ndarray,
    weights: np.ndarray,
    points: np.ndarray,
    C: float,
    *,
    kind: str = "full",
    J: int = 6,
    block_size: int = 2048,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute an electromagnetic DSM indicator directly from far-field data.

    The input data is the raw polarimetric channel vector
    ``g_i = M(p_i)c(p_i)`` at each Fourier node.  The indicator uses the
    adjoint electromagnetic test matrix,

        I(z) = ||sum_i w_i M(p_i)^* g_i exp(i C p_i.z)||_2.

    Unlike :func:`direct_sampling_tensor_indicator`, this does not first
    recover the tensor Fourier coefficients ``c(p_i)`` by pseudo-inversion.
    It is a direct matched-filter/backprojection using the Maxwell far-field
    channel structure, including observation projection and incident
    polarizations.
    """
    from common.polarimetric import build_polarimetric_matrix

    data = np.asarray(farfield_data, dtype=np.complex128)
    nodes = np.asarray(p_nodes, dtype=float)
    if data.ndim != 2:
        raise ValueError("farfield_data must have shape (n_nodes, 6*J)")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    if data.shape != (nodes.shape[0], 6 * int(J)):
        raise ValueError("farfield_data must have shape (n_nodes, 6*J)")

    adjoint_data = []
    for p, datum in zip(nodes, data):
        M = build_polarimetric_matrix(p, kind=kind, J=J)
        adjoint_data.append(M.conj().T @ datum)
    adjoint_data_arr = np.asarray(adjoint_data, dtype=np.complex128)

    image, meta = direct_sampling_tensor_indicator(
        adjoint_data_arr,
        nodes,
        weights,
        points,
        C,
        block_size=block_size,
        normalize=normalize,
    )
    meta = {
        "em_dsm_nodes": int(np.asarray(p_nodes).shape[0]),
        "em_dsm_channels": int(data.shape[1]) if data.ndim == 2 else 0,
        "em_dsm_adjoint_components": int(adjoint_data_arr.shape[1]) if adjoint_data_arr.ndim == 2 else 0,
        "em_dsm_normalized": int(bool(normalize)),
        "em_dsm_raw_max": float(meta["dsm_raw_max"]),
        "em_dsm_raw_l2": float(meta["dsm_raw_l2"]),
        "farfield_data_norm": vector_norm(data),
        "farfield_data_max_abs": float(np.max(np.abs(data))) if data.size else float("nan"),
        "farfield_adjoint_data_norm": float(meta["data_norm"]),
        "farfield_adjoint_data_max_abs": float(meta["data_max_abs"]),
        "farfield_weighted_norm": float(meta["coeff_norm"]),
        "farfield_weighted_max_abs": float(meta["coeff_max_abs"]),
    }
    return image, meta
