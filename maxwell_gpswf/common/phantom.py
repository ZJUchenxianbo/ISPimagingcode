#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phantom (scatterer) definitions and tensor basis utilities.

All scatterer geometry and contrast definitions live here.  To add a new
phantom, define a factory function that returns ``list[Block]`` and/or the
corresponding tensor contrast ``Q(x)`` on voxel grids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
TensorKind = Literal["full", "reciprocal", "isotropic"]


@dataclass(frozen=True)
class Block:
    """Axis-aligned rectangular block scatterer."""

    center: tuple[float, float, float]
    half_width: tuple[float, float, float]
    amplitude: complex


@dataclass(frozen=True)
class TensorBlock:
    """Axis-aligned block scatterer with its own 3x3 tensor contrast."""

    center: tuple[float, float, float]
    half_width: tuple[float, float, float]
    tensor: CArray


@dataclass(frozen=True)
class Mode:
    """Ball-GPSWF mode identifier with precomputed data."""

    ell: int
    n: int
    m: int
    alpha: complex
    beta: np.ndarray


# ---------------------------------------------------------------------------
# Phantom factories
# ---------------------------------------------------------------------------


def three_block_phantom(variant: str = "born") -> list[Block]:
    """Three-block phantom with minimum boundary separation 0.20.

    Parameters
    ----------
    variant : str
        ``"born"`` — amplitudes matching the main Born reconstruction experiment.
        ``"vie"``  — larger contrast amplitudes used in the VIE comparison.
    """
    if variant == "born":
        return [
            Block(center=(-0.26, 0.25, 0.0), half_width=(0.16, 0.16, 0.16), amplitude=1.00 + 0.10j),
            Block(center=(0.25, 0.25, 0.0), half_width=(0.15, 0.15, 0.16), amplitude=0.85 + 0.05j),
            Block(center=(0.00, -0.25, 0.0), half_width=(0.18, 0.14, 0.16), amplitude=1.15 - 0.05j),
        ]
    if variant == "vie":
        return [
            Block(center=(-0.26, 0.25, 0.0), half_width=(0.16, 0.16, 0.16), amplitude=0.60 + 0.05j),
            Block(center=(0.25, 0.25, 0.0), half_width=(0.15, 0.15, 0.16), amplitude=1.15 + 0.10j),
            Block(center=(0.00, -0.25, 0.0), half_width=(0.18, 0.14, 0.16), amplitude=1.80 - 0.10j),
        ]
    raise ValueError(f"Unknown three_block_phantom variant: {variant!r}")


def three_tensor_block_phantom() -> list[TensorBlock]:
    """Three gap-0.20 blocks with different anisotropic tensor contrasts."""
    return [
        TensorBlock(
            center=(-0.26, 0.25, 0.0),
            half_width=(0.16, 0.16, 0.16),
            tensor=np.asarray(
                [
                    [1.00 + 0.15j, 0.32 - 0.18j, -0.08 + 0.06j],
                    [-0.22 + 0.10j, -0.50 + 0.24j, 0.18 - 0.12j],
                    [0.12 + 0.03j, -0.26 + 0.08j, 0.65 - 0.25j],
                ],
                dtype=np.complex128,
            ),
        ),
        TensorBlock(
            center=(0.25, 0.25, 0.0),
            half_width=(0.15, 0.15, 0.16),
            tensor=np.asarray(
                [
                    [0.25 + 0.05j, -0.18 + 0.08j, 0.42 - 0.10j],
                    [0.30 - 0.16j, 1.05 + 0.12j, -0.20 + 0.18j],
                    [-0.15 + 0.04j, 0.36 - 0.07j, -0.45 + 0.22j],
                ],
                dtype=np.complex128,
            ),
        ),
        TensorBlock(
            center=(0.00, -0.25, 0.0),
            half_width=(0.18, 0.14, 0.16),
            tensor=np.asarray(
                [
                    [-0.55 + 0.28j, 0.16 + 0.12j, 0.12 - 0.20j],
                    [-0.08 + 0.05j, 0.35 - 0.18j, 0.48 + 0.06j],
                    [0.24 - 0.10j, -0.30 + 0.16j, 1.25 - 0.30j],
                ],
                dtype=np.complex128,
            ),
        ),
    ]


def ball_phantom(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.35,
    amplitude: complex = 1.0 + 0.0j,
) -> list[Block]:
    """Single ball phantom approximated by a cube block."""
    return [Block(center=center, half_width=(radius, radius, radius), amplitude=amplitude)]


# ---------------------------------------------------------------------------
# Analytical Fourier profile (Born approximation)
# ---------------------------------------------------------------------------


def block_fourier_profile(p_nodes: np.ndarray, blocks: list[Block], C: float) -> np.ndarray:
    """Analytical Fourier transform of block phantom under ``exp(-i C p·x)``.

    This is the **unified Fourier convention** for the entire project.
    Each rectangular block contributes a sinc-factor product.
    """
    xi = float(C) * np.asarray(p_nodes, dtype=float)
    values = np.zeros(p_nodes.shape[0], dtype=np.complex128)
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        phase = np.exp(-1j * (xi @ center))
        volume_profile = np.prod(2.0 * half_width * np.sinc((xi * half_width) / math.pi), axis=1)
        values += complex(block.amplitude) * phase * volume_profile
    return values


def tensor_block_fourier_coefficients(
    p_nodes: np.ndarray,
    blocks: list[TensorBlock],
    C: float,
    kind: TensorKind = "full",
) -> CArray:
    """Analytical Fourier coefficients for a tensor-block phantom.

    Returns an array with shape ``(n_nodes, n_tensor_coeffs)`` representing
    ``Qhat(p) = sum_r c_r(p) T_r`` in the selected tensor basis.
    """
    p_nodes = np.asarray(p_nodes, dtype=float)
    if p_nodes.ndim != 2 or p_nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    n_coeffs = len(tensor_basis(kind))
    values = np.zeros((p_nodes.shape[0], n_coeffs), dtype=np.complex128)
    xi = float(C) * p_nodes
    for block in blocks:
        center = np.asarray(block.center, dtype=float)
        half_width = np.asarray(block.half_width, dtype=float)
        phase = np.exp(-1j * (xi @ center))
        profile = np.prod(2.0 * half_width * np.sinc((xi * half_width) / math.pi), axis=1)
        tensor_coeffs = tensor_coefficients_from_matrix(block.tensor, kind)
        values += (phase * profile)[:, None] * tensor_coeffs[None, :]
    return values


def truth_image_2d(
    grid_size: int, blocks: list[Block], component_value: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the z=0 cross-section ground truth image.

    Returns
    -------
    truth : (grid_size, grid_size) complex array
    grid_points : (grid_size*grid_size, 3) float array (z=0)
    disk_mask : (grid_size, grid_size) bool array
    """
    centers = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(centers, centers)
    truth = np.zeros((grid_size, grid_size), dtype=np.complex128)
    for block in blocks:
        cx, cy, cz = block.center
        hx, hy, hz = block.half_width
        if abs(cz) <= hz:
            mask = (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
            truth[mask] += complex(block.amplitude) * component_value
    disk_mask = X**2 + Y**2 <= 1.0
    truth[~disk_mask] = 0.0
    grid_points = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
    return truth, grid_points, disk_mask


def tensor_truth_image_2d(
    grid_size: int,
    blocks: list[TensorBlock],
    *,
    kind: TensorKind = "full",
    component_index: int | None = None,
    display: Literal["frobenius", "component"] = "frobenius",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a z=0 truth image for tensor-block phantoms."""
    centers = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(centers, centers)
    matrix_field = np.zeros((grid_size, grid_size, 3, 3), dtype=np.complex128)
    for block in blocks:
        cx, cy, cz = block.center
        hx, hy, hz = block.half_width
        if abs(cz) <= hz:
            mask = (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
            matrix_field[mask] += np.asarray(block.tensor, dtype=np.complex128)

    if display == "frobenius":
        truth = np.linalg.norm(matrix_field.reshape(grid_size, grid_size, 9), axis=2)
        truth = truth.astype(np.complex128)
    elif display == "component":
        if component_index is None:
            raise ValueError("component_index is required for component display")
        coeff_field = np.zeros((grid_size, grid_size, len(tensor_basis(kind))), dtype=np.complex128)
        for i in range(grid_size):
            for j in range(grid_size):
                coeff_field[i, j] = tensor_coefficients_from_matrix(matrix_field[i, j], kind)
        truth = coeff_field[:, :, int(component_index)]
    else:
        raise ValueError("display must be 'frobenius' or 'component'")

    disk_mask = X**2 + Y**2 <= 1.0
    truth[~disk_mask] = 0.0
    grid_points = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
    return truth, grid_points, disk_mask


# ---------------------------------------------------------------------------
# Tensor basis and reference tensors
# ---------------------------------------------------------------------------


def tensor_basis(kind: TensorKind) -> list[Array]:
    """Return real tensor bases used for the polarimetric matrix."""
    if kind == "full":
        basis = []
        for a in range(3):
            for b in range(3):
                T = np.zeros((3, 3), dtype=float)
                T[a, b] = 1.0
                basis.append(T)
        return basis
    if kind == "reciprocal":
        basis = []
        for a, b in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
            T = np.zeros((3, 3), dtype=float)
            scale = 1.0 if a == b else 1.0 / math.sqrt(2.0)
            T[a, b] = scale
            T[b, a] = scale
            basis.append(T)
        return basis
    if kind == "isotropic":
        return [np.eye(3)]
    raise ValueError("kind must be 'full', 'reciprocal', or 'isotropic'")


def tensor_coefficients_from_matrix(T: CArray, kind: TensorKind) -> CArray:
    """Return coordinates of a tensor matrix in the chosen tensor basis."""
    T = np.asarray(T, dtype=np.complex128)
    if T.shape != (3, 3):
        raise ValueError("T must have shape (3, 3)")
    if kind == "full":
        return np.asarray([T[a, b] for a in range(3) for b in range(3)], dtype=np.complex128)
    if kind == "reciprocal":
        return np.asarray(
            [
                T[0, 0],
                T[1, 1],
                T[2, 2],
                math.sqrt(2.0) * T[0, 1],
                math.sqrt(2.0) * T[0, 2],
                math.sqrt(2.0) * T[1, 2],
            ],
            dtype=np.complex128,
        )
    if kind == "isotropic":
        return np.asarray([np.trace(T) / 3.0], dtype=np.complex128)
    raise ValueError("kind must be 'full', 'reciprocal', or 'isotropic'")


def reference_tensor(kind: TensorKind) -> CArray:
    """Return a fixed tensor phantom compatible with the chosen tensor class."""
    if kind == "isotropic":
        return (1.0 + 0.35j) * np.eye(3, dtype=np.complex128)
    if kind == "reciprocal":
        return np.asarray(
            [
                [1.00 + 0.20j, 0.25 - 0.10j, -0.18 + 0.05j],
                [0.25 - 0.10j, -0.55 + 0.30j, 0.32 + 0.12j],
                [-0.18 + 0.05j, 0.32 + 0.12j, 0.80 - 0.25j],
            ],
            dtype=np.complex128,
        )
    if kind == "full":
        return np.asarray(
            [
                [1.00 + 0.15j, 0.35 - 0.20j, -0.10 + 0.08j],
                [-0.28 + 0.12j, -0.65 + 0.30j, 0.22 - 0.18j],
                [0.16 + 0.04j, -0.34 + 0.10j, 0.75 - 0.35j],
            ],
            dtype=np.complex128,
        )
    raise ValueError("kind must be 'full', 'reciprocal', or 'isotropic'")


def normalized_ball_fourier_profile(p_nodes: Array, C: float, radius: float) -> CArray:
    """Fourier transform profile of a constant ball, normalized to value 1 at zero.

    The unnormalized integral is
    ``4*pi*(sin(z)-z*cos(z))/|xi|^3`` with ``z=|xi|*radius``.
    """
    p_nodes = np.asarray(p_nodes, dtype=float)
    if p_nodes.ndim != 2 or p_nodes.shape[1] != 3:
        raise ValueError("p_nodes must have shape (n_nodes, 3)")
    z = float(C) * float(radius) * np.linalg.norm(p_nodes, axis=1)
    values = np.empty_like(z, dtype=np.complex128)
    small = np.abs(z) < 1e-8
    values[small] = 1.0 + 0.0j
    zz = z[~small]
    values[~small] = 3.0 * (np.sin(zz) - zz * np.cos(zz)) / (zz**3)
    return values


# ---------------------------------------------------------------------------
# Additional phantoms for Figure 3 (sphere, cube, combinations)
# ---------------------------------------------------------------------------


def cube_phantom(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    half_side: float = 0.2,
    amplitude: complex = 1.0 + 0.0j,
) -> list[Block]:
    """Single cube phantom."""
    hs = (half_side, half_side, half_side)
    return [Block(center=center, half_width=hs, amplitude=amplitude)]


def two_spheres_cube_phantom() -> list[Block]:
    """Two spheres + one cube combination.

    Spheres are approximated by cubes for the Block-based framework.
    """
    return [
        Block(center=(-0.35, 0.25, 0.0), half_width=(0.14, 0.14, 0.14), amplitude=1.00 + 0.05j),
        Block(center=(0.30, 0.20, 0.0), half_width=(0.12, 0.12, 0.12), amplitude=0.80 - 0.10j),
        Block(center=(0.00, -0.30, 0.0), half_width=(0.18, 0.18, 0.18), amplitude=1.20 + 0.10j),
    ]


def dispersed_blocks_phantom() -> list[Block]:
    """Several small dispersed blocks at irregular positions."""
    return [
        Block(center=(-0.55, 0.40, 0.0), half_width=(0.08, 0.10, 0.08), amplitude=0.90 + 0.05j),
        Block(center=(0.45, 0.45, 0.0), half_width=(0.07, 0.07, 0.07), amplitude=1.10 - 0.05j),
        Block(center=(-0.20, -0.50, 0.0), half_width=(0.10, 0.08, 0.10), amplitude=0.70 + 0.10j),
        Block(center=(0.55, -0.30, 0.0), half_width=(0.09, 0.11, 0.09), amplitude=1.30 - 0.05j),
        Block(center=(-0.50, -0.10, 0.0), half_width=(0.06, 0.09, 0.06), amplitude=0.60 + 0.00j),
        Block(center=(0.10, 0.05, 0.0), half_width=(0.12, 0.06, 0.12), amplitude=0.85 + 0.08j),
    ]


# ---------------------------------------------------------------------------
# Sphere: analytical Fourier profile and circular cross-section truth
# ---------------------------------------------------------------------------


def sphere_fourier_profile(
    p_nodes: Array, center: Array, radius: float, amplitude: complex, C: float,
) -> CArray:
    """Analytical Fourier transform of a constant ball.

    ``Q̂(ξ) = 4π (sin(z) - z cos(z)) / ξ³ * amplitude``
    where ``z = |ξ| * radius``, ``ξ = C * p``, and the phase factor
    ``exp(-i ξ · center)`` accounts for the offset.
    """
    p = np.asarray(p_nodes, dtype=float)
    center = np.asarray(center, dtype=float)
    xi = float(C) * p
    xi_norm = np.linalg.norm(xi, axis=1)
    z = xi_norm * float(radius)
    profile = np.empty_like(z, dtype=np.complex128)
    small = z < 1e-8
    profile[small] = (4.0 * math.pi / 3.0) * float(radius) ** 3
    profile[~small] = 4.0 * math.pi * (np.sin(z[~small]) - z[~small] * np.cos(z[~small])) / (xi_norm[~small] ** 3)
    phase = np.exp(-1j * (xi @ center))
    return complex(amplitude) * phase * profile


def sphere_truth_2d(
    grid_size: int, center: tuple[float, float, float],
    radius: float, amplitude: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular z=0 cross-section of a sphere.

    Returns (truth, grid_points, disk_mask) — same convention as
    :func:`truth_image_2d`.
    """
    cx, cy, cz = center
    centers = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(centers, centers)
    truth = np.zeros((grid_size, grid_size), dtype=np.complex128)
    # Sphere intersects z=0 if |cz| <= radius
    if abs(cz) <= radius:
        r_cut = math.sqrt(max(radius**2 - cz**2, 0.0))
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r_cut**2
        truth[mask] = complex(amplitude)
    disk_mask = X**2 + Y**2 <= 1.0
    truth[~disk_mask] = 0.0
    grid_points = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
    return truth, grid_points, disk_mask


def _shape_truth_and_fourier(
    shape_name: str, p_nodes: np.ndarray, grid_size: int, C: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: return (truth, grid_points, disk_mask, fourier_data) for a shape."""
    if shape_name == "sphere":
        center = (0.0, 0.0, 0.0)
        radius = 0.25
        amp = 1.0 + 0.0j
        truth, gp, dm = sphere_truth_2d(grid_size, center, radius, amp)
        fourier = sphere_fourier_profile(p_nodes, np.array(center), radius, amp, C)
        return truth, gp, dm, fourier
    elif shape_name == "cube":
        blocks = cube_phantom(center=(0.0, 0.0, 0.0), half_side=0.2, amplitude=1.0 + 0.0j)
        comp_val = tensor_coefficients_from_matrix(reference_tensor("isotropic"), "isotropic")[0]
        truth, gp, dm = truth_image_2d(grid_size, blocks, comp_val)
        fourier = block_fourier_profile(p_nodes, blocks, C) * comp_val
        return truth, gp, dm, fourier
    elif shape_name == "two_spheres_cube":
        blocks = two_spheres_cube_phantom()
        comp_val = tensor_coefficients_from_matrix(reference_tensor("isotropic"), "isotropic")[0]
        truth, gp, dm = truth_image_2d(grid_size, blocks, comp_val)
        fourier = block_fourier_profile(p_nodes, blocks, C) * comp_val
        return truth, gp, dm, fourier
    elif shape_name == "dispersed":
        blocks = dispersed_blocks_phantom()
        comp_val = tensor_coefficients_from_matrix(reference_tensor("isotropic"), "isotropic")[0]
        truth, gp, dm = truth_image_2d(grid_size, blocks, comp_val)
        fourier = block_fourier_profile(p_nodes, blocks, C) * comp_val
        return truth, gp, dm, fourier
    elif shape_name == "inhomogeneous":
        fourier = _inhomogeneous_fourier(p_nodes, C)
        truth, gp, dm = _inhomogeneous_truth(grid_size)
        return truth, gp, dm, fourier
    else:
        raise ValueError(f"Unknown shape: {shape_name!r}")


# ---------------------------------------------------------------------------
# Inhomogeneous medium: sum of Gaussian bumps inside a disk
# ---------------------------------------------------------------------------

def _inhomogeneous_fourier(p_nodes, C):
    """Fourier transform of a sum of Gaussian bumps (analytical)."""
    xi = float(C) * np.asarray(p_nodes, dtype=float)
    xi_sq = np.sum(xi * xi, axis=1)
    total = np.zeros(p_nodes.shape[0], dtype=np.complex128)
    # Three Gaussian bumps with different centers, widths, amplitudes
    bumps = [
        (np.array([-0.20, 0.15, 0.0]), 0.12, 1.0 + 0.0j),
        (np.array([0.25, 0.10, 0.0]), 0.08, 0.7 + 0.2j),
        (np.array([0.05, -0.25, 0.0]), 0.14, 1.2 - 0.1j),
    ]
    for center, sigma, amp in bumps:
        # Q̂(ξ) = (2π)^(3/2) * σ³ * exp(-σ²|ξ|²/2) * exp(-i ξ·center)
        prefactor = (2.0 * math.pi) ** 1.5 * sigma**3
        fourier = prefactor * np.exp(-0.5 * sigma**2 * xi_sq)
        phase = np.exp(-1j * (xi @ center))
        total += complex(amp) * phase * fourier
    return total


def _inhomogeneous_truth(grid_size):
    """z=0 cross-section of Gaussian bumps."""
    xs = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(xs, xs)
    truth = np.zeros((grid_size, grid_size), dtype=np.complex128)
    bumps = [
        (np.array([-0.20, 0.15, 0.0]), 0.12, 1.0 + 0.0j),
        (np.array([0.25, 0.10, 0.0]), 0.08, 0.7 + 0.2j),
        (np.array([0.05, -0.25, 0.0]), 0.14, 1.2 - 0.1j),
    ]
    for center, sigma, amp in bumps:
        r_sq = (X - center[0])**2 + (Y - center[1])**2 + (0 - center[2])**2
        truth += complex(amp) * np.exp(-0.5 * r_sq / sigma**2)
    disk_mask = X**2 + Y**2 <= 1.0
    truth[~disk_mask] = 0.0
    grid_points = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(grid_size * grid_size)])
    return truth, grid_points, disk_mask
