#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反散射实验中可复用的合成目标案例。

当前按研究对象分为三类：

1. 不可穿透声软障碍体 ``impenetrable_obstacle_cases``：
   - ``small_target``：一个靠近原点的小星形障碍物。
   - ``small_cluster``：三个小星形障碍物，用于观察分辨率。
   - ``large_target``：一个较大的星形障碍物，用于比较尺寸效应。

2. 可穿透介质 ``medium_cases``：
   - ``homogeneous_disk``：圆盘支撑上的均匀折射率/对比度介质。
   - ``inhomogeneous_disk``：圆盘支撑上的非均匀介质，带低阶空间调制。
   这些 cases 目前只定义目标参数，供后续介质散射前向/反演脚本使用。

3. 点散射体 ``point_scatterer_cases``：
   - ``three_point_scatterers``：三个各向同性点散射体，带复强度。

命令行脚本可通过 ``--cases`` 或 ``--case`` 使用这些名称。旧别名
``one_small``、``three_small``、``one_large`` 只用于兼容早期输出名称。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy as np

from common.scattering import PI2, Array, CArray

CaseT = TypeVar("CaseT")


@dataclass(frozen=True)
class ObstacleTargetCase:
    """不可穿透障碍体目标案例。"""
    name: str
    label: str
    params: Array
    n_obstacles: int
    n_boundary: int
    grid_extent: float


@dataclass(frozen=True)
class MediumTargetCase:
    """可穿透介质目标案例。"""
    name: str
    label: str
    contrast_profile: str
    center: tuple[float, float]
    radius: float
    contrast_params: tuple[tuple[str, float], ...]
    grid_extent: float


@dataclass(frozen=True)
class PointScattererCase:
    """点散射体目标案例。"""
    name: str
    label: str
    points: Array
    strengths: CArray
    grid_extent: float


def make_star_block(
    center: tuple[float, float],
    radius: float,
    a2c: float = 0.0,
    a2s: float = 0.0,
    a3c: float = 0.0,
    a3s: float = 0.0,
) -> Array:
    """Build a seven-parameter star-shaped obstacle block."""
    return np.array([center[0], center[1], radius, a2c, a2s, a3c, a3s], dtype=float)


# ---------------------------------------------------------------------------
# 障碍物几何 — 星形边界参数化与离散化
# ---------------------------------------------------------------------------

@dataclass
class BoundaryGeometry:
    """Boundary discretization data for one or more obstacles."""
    x: Array
    normal: Array
    ds: Array
    obs_id: Array  # int array


def star_radius(theta: Array, r0: float, a2c: float, a2s: float, a3c: float, a3s: float) -> Array:
    """Radial function r(theta) for a star-like obstacle."""
    return r0 * (
        1.0
        + a2c * np.cos(2.0 * theta)
        + a2s * np.sin(2.0 * theta)
        + a3c * np.cos(3.0 * theta)
        + a3s * np.sin(3.0 * theta)
    )


def star_radius_derivative(theta: Array, r0: float, a2c: float, a2s: float, a3c: float, a3s: float) -> Array:
    """Derivative of r(theta) with respect to theta."""
    return r0 * (
        -2.0 * a2c * np.sin(2.0 * theta)
        + 2.0 * a2s * np.cos(2.0 * theta)
        - 3.0 * a3c * np.sin(3.0 * theta)
        + 3.0 * a3s * np.cos(3.0 * theta)
    )


def star_boundary(center: tuple[float, float], coeffs: Array, n_pts: int) -> tuple[Array, Array, Array]:
    """Discretize a star-like boundary into points, normals, and arclength weights."""
    r0, a2c, a2s, a3c, a3s = [float(v) for v in coeffs]
    t = np.linspace(0.0, PI2, n_pts, endpoint=False)
    r = star_radius(t, r0, a2c, a2s, a3c, a3s)
    rp = star_radius_derivative(t, r0, a2c, a2s, a3c, a3s)
    ct = np.cos(t)
    st = np.sin(t)
    x = np.column_stack([center[0] + r * ct, center[1] + r * st])

    dx = rp * ct - r * st
    dy = rp * st + r * ct
    speed = np.sqrt(dx * dx + dy * dy)
    ds = speed * (PI2 / n_pts)
    normal = np.column_stack([dy / speed, -dx / speed])
    return x, normal, ds


def obstacle_param_slice(j: int) -> slice:
    """Return the slice for obstacle j in a concatenated parameter vector."""
    return slice(7 * j, 7 * (j + 1))


def params_to_geometry(params: Array, n_per_obstacle: int, n_obstacles: int = 3) -> BoundaryGeometry:
    """Convert concatenated obstacle parameters to boundary geometry."""
    params = np.asarray(params, dtype=float)
    if n_obstacles <= 0:
        raise ValueError("n_obstacles must be positive")
    if n_per_obstacle <= 0:
        raise ValueError("n_per_obstacle must be positive")
    expected_size = 7 * n_obstacles
    if params.size != expected_size:
        raise ValueError(f"params must contain {expected_size} values for {n_obstacles} obstacle(s)")

    xs: list[Array] = []
    normals: list[Array] = []
    dss: list[Array] = []
    ids: list[Array] = []
    for j in range(n_obstacles):
        block = params[obstacle_param_slice(j)]
        center = (float(block[0]), float(block[1]))
        coeffs = block[2:7]
        x, nrm, ds = star_boundary(center, coeffs, n_per_obstacle)
        xs.append(x)
        normals.append(nrm)
        dss.append(ds)
        ids.append(np.full(n_per_obstacle, j, dtype=int))
    return BoundaryGeometry(
        x=np.vstack(xs),
        normal=np.vstack(normals),
        ds=np.concatenate(dss),
        obs_id=np.concatenate(ids).astype(np.int64, copy=False),
    )


def dense_boundary_points(params_obs: Array, n: int = 400) -> Array:
    """Generate dense boundary points for plotting."""
    center = (float(params_obs[0]), float(params_obs[1]))
    return star_boundary(center, params_obs[2:7], n)[0]


def plot_obstacle_boundaries(
    ax: Any,
    params: Array,
    n_obstacles: int,
    style: str = "k--",
    *,
    lw: float = 1.2,
    label: str | None = None,
    n: int = 400,
    **plot_kwargs: Any,
) -> None:
    """Plot one or more parameterized obstacle boundaries on an axes."""
    for j in range(n_obstacles):
        pts = dense_boundary_points(params[obstacle_param_slice(j)], n=n)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            style,
            lw=lw,
            label=label if label is not None and j == 0 else None,
            **plot_kwargs,
        )


def deduplicate_legend(ax: Any, loc: str = "best") -> None:
    """Show one legend entry per label on an axes."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc=loc)


# ---------------------------------------------------------------------------
# 目标案例定义
# ---------------------------------------------------------------------------


def _impenetrable_obstacle_case_map() -> dict[str, ObstacleTargetCase]:
    """Return available impenetrable sound-soft obstacle cases."""
    small_target = np.concatenate([
        make_star_block((0.02, -0.01), 0.045, 0.10, -0.06, 0.04, 0.02),
    ])

    small_cluster = np.concatenate([
        make_star_block((-0.16, -0.08), 0.043, 0.10, -0.05, 0.04, 0.02),
        make_star_block((0.12, -0.02), 0.046, -0.08, 0.08, -0.04, 0.03),
        make_star_block((0.00, 0.15), 0.044, 0.06, 0.09, 0.04, -0.05),
    ])

    large_target = np.concatenate([
        make_star_block((0.02, 0.00), 0.18, 0.22, -0.08, 0.07, 0.05),
    ])

    return {
        "small_target": ObstacleTargetCase("small_target", "small target", small_target, 1, 80, 0.45),
        "small_cluster": ObstacleTargetCase("small_cluster", "small-target cluster", small_cluster, 3, 48, 0.50),
        "large_target": ObstacleTargetCase("large_target", "large target", large_target, 1, 180, 0.60),
        # Backward-compatible aliases for earlier output names and notes.
        "one_small": ObstacleTargetCase("one_small", "1 small target", small_target, 1, 80, 0.45),
        "three_small": ObstacleTargetCase("three_small", "3 small targets", small_cluster, 3, 48, 0.50),
        "one_large": ObstacleTargetCase("one_large", "1 large target", large_target, 1, 180, 0.60),
    }


def _medium_case_map() -> dict[str, MediumTargetCase]:
    """Return available penetrable-medium cases."""
    return {
        "homogeneous_disk": MediumTargetCase(
            name="homogeneous_disk",
            label="homogeneous disk medium",
            contrast_profile="homogeneous",
            center=(0.0, 0.0),
            radius=0.18,
            contrast_params=(("contrast", 1.0),),
            grid_extent=0.55,
        ),
        "inhomogeneous_disk": MediumTargetCase(
            name="inhomogeneous_disk",
            label="inhomogeneous disk medium",
            contrast_profile="inhomogeneous",
            center=(0.0, 0.0),
            radius=0.20,
            contrast_params=(
                ("base_contrast", 0.8),
                ("modulation_amplitude", 0.35),
                ("mode_x", 2.0),
                ("mode_y", 1.0),
            ),
            grid_extent=0.60,
        ),
    }


def _point_scatterer_case_map() -> dict[str, PointScattererCase]:
    """Return available point-scatterer cases."""
    points = np.array(
        [
            [-0.20, -0.10],
            [0.30, -0.10],
            [-0.20, 0.20],
        ],
        dtype=float,
    )
    strengths = np.array([1.0 + 0.0j, 0.85 * np.exp(0.4j), 1.15 * np.exp(-0.7j)], dtype=complex)
    return {
        "three_point_scatterers": PointScattererCase(
            name="three_point_scatterers",
            label="three point scatterers",
            points=points,
            strengths=strengths,
            grid_extent=0.42,
        )
    }


def parse_case_names(text: str) -> list[str] | None:
    """Parse a comma-separated case list; return None for all default cases."""
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names or any(name.lower() == "all" for name in names):
        return None
    return names


def _select_cases(
    cases: dict[str, CaseT],
    default_names: Sequence[str],
    case_names: Sequence[str] | None,
    category_label: str,
) -> list[CaseT]:
    """Select named cases and report a useful error for unknown names."""
    selected = default_names if case_names is None else list(case_names)
    missing = [name for name in selected if name not in cases]
    if missing:
        available = ", ".join(default_names)
        aliases = ", ".join(sorted(set(cases) - set(default_names)))
        alias_text = f"; aliases: {aliases}" if aliases else ""
        raise ValueError(
            f"unknown {category_label} case(s): {', '.join(missing)}; "
            f"available: {available}{alias_text}"
        )
    return [cases[name] for name in selected]


def impenetrable_obstacle_cases(case_names: Sequence[str] | None = None) -> list[ObstacleTargetCase]:
    """Return selected impenetrable obstacle cases."""
    cases = _impenetrable_obstacle_case_map()
    default_names = ["small_target", "small_cluster", "large_target"]
    return _select_cases(cases, default_names, case_names, "obstacle")


def limited_aperture_obstacle_cases(case_names: Sequence[str] | None = None) -> list[ObstacleTargetCase]:
    """Backward-compatible name for obstacle cases used by limited-aperture scripts."""
    return impenetrable_obstacle_cases(case_names)


def medium_cases(case_names: Sequence[str] | None = None) -> list[MediumTargetCase]:
    """Return selected penetrable-medium cases."""
    cases = _medium_case_map()
    default_names = ["homogeneous_disk", "inhomogeneous_disk"]
    return _select_cases(cases, default_names, case_names, "medium")


def point_scatterer_cases(case_names: Sequence[str] | None = None) -> list[PointScattererCase]:
    """Return selected point-scatterer cases."""
    cases = _point_scatterer_case_map()
    default_names = ["three_point_scatterers"]
    return _select_cases(cases, default_names, case_names, "point-scatterer")
