#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytical Born approximation for Maxwell far-field data.

All functions in this module use the **unified Fourier convention**
``exp(-i C p·x)`` where ``C = 2k`` and ``p = (d - x̂)/2``.

The VIE-based Born far field (in ``forward/vie.py``) uses the opposite
sign convention; convert with :func:`forward.vie.vie_to_fourier_convention`.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def _unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-14:
        raise ValueError("vector must be nonzero")
    return v / norm


def maxwell_born_far_field_fourier_formula(
    nodes: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    k: float,
    direction: np.ndarray,
    polarization: np.ndarray,
    obs_dirs: np.ndarray,
    *,
    phase_sign: int = -1,
) -> np.ndarray:
    """Born far field from the tensor Fourier sum.

    Uses the unified convention by default (``phase_sign=-1`` →
    ``exp(-i k (d - x̂)·y)``).

    ``phase_sign=+1`` is provided for diagnostic comparison with the VIE
    Born path (which uses ``exp(+i k (d - x̂)·y)`` internally).
    """
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    Q = np.asarray(Q, dtype=np.complex128)
    direction = _unit_vector(direction)
    polarization = np.asarray(polarization, dtype=np.complex128)
    obs_dirs = np.asarray(obs_dirs, dtype=float)
    if Q.shape != (nodes.shape[0], 3, 3):
        raise ValueError("Q must have shape (n_nodes, 3, 3)")

    farfield = np.zeros((obs_dirs.shape[0], 3), dtype=np.complex128)
    for idx, obs in enumerate(obs_dirs):
        xhat = _unit_vector(obs)
        xi = float(k) * (direction - xhat)
        phase = np.exp(1j * int(phase_sign) * (nodes @ xi))
        tensor_hat = np.einsum("n,nij->ij", weights * phase, Q)
        projector = np.eye(3) - np.outer(xhat, xhat)
        farfield[idx] = (float(k) ** 2 / (4.0 * math.pi)) * (projector @ tensor_hat @ polarization)
    return farfield
