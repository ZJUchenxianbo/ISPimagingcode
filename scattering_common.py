#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants, type aliases, and utilities for 2D scattering experiments.

This module is the dependency root of the project.  It must not import from any
other project module.  All foundation modules may import from here.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 数学常量
# ---------------------------------------------------------------------------
PI2 = 2.0 * math.pi

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

# ---------------------------------------------------------------------------
# 方向向量
# ---------------------------------------------------------------------------

def direction_vectors(angles: Array) -> Array:
    """Map polar angles to unit vectors (cos(theta), sin(theta))."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1:
        raise ValueError("angles must be one-dimensional")
    return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------------------------------------------------------------------------
# 命令行解析
# ---------------------------------------------------------------------------

def parse_float_list(text: str) -> Array:
    """Parse a comma-separated string into a float array."""
    vals = [float(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# 噪声与信噪比
# ---------------------------------------------------------------------------

def add_relative_noise(
    data: CArray,
    rel_noise: float,
    rng_or_seed: np.random.Generator | int | None = None,
) -> CArray:
    """Add complex Gaussian noise scaled to rel_noise * ||data||_2.

    ``rng_or_seed`` accepts a ``numpy.random.Generator``, an integer seed, or
    ``None`` (fresh entropy).
    """
    data = np.asarray(data, dtype=np.complex128)
    if rel_noise <= 0.0:
        return data.copy()
    rng = (
        rng_or_seed
        if isinstance(rng_or_seed, np.random.Generator)
        else np.random.default_rng(rng_or_seed)
    )
    noise = rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
    noise_norm = max(float(np.linalg.norm(noise)), 1e-14)
    amp = float(rel_noise) * float(np.linalg.norm(data))
    return data + amp * noise / noise_norm


def empirical_snr(clean: CArray, noisy: CArray) -> float:
    """Return empirical SNR ||clean||_2 / ||noisy - clean||_2."""
    return float(
        np.linalg.norm(clean) / max(np.linalg.norm(noisy - clean), 1e-14)
    )


# ---------------------------------------------------------------------------
# 文件名辅助
# ---------------------------------------------------------------------------

def safe_slug(parts: Iterable[object]) -> str:
    """Build a conservative filename slug from display-label parts."""
    text = "_".join(str(part) for part in parts)
    chars = []
    for ch in text.strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"
