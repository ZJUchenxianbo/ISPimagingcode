"""Independent interface geometry and transpose-consistent acquisition weights."""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Geometry:
    top: float
    nodes: tuple
    bottom: tuple

    def evaluate(self, x):
        return np.interp(x, self.nodes, self.bottom)

    def as_dict(self):
        return asdict(self)

    def validate(self, config):
        nodes, bottom = np.asarray(self.nodes), np.asarray(self.bottom)
        if nodes.ndim != 1 or len(nodes) < 2 or bottom.shape != nodes.shape:
            raise ValueError("Geometry needs matching one-dimensional nodes and bottom")
        if not np.isfinite(nodes).all() or not np.isfinite(bottom).all() or not np.isfinite(self.top):
            raise ValueError("Nonfinite geometry")
        if np.any(np.diff(nodes) <= 0):
            raise ValueError("Geometry nodes must be increasing")
        margin = config.interface_width_cells * config.grid_spacing
        if self.top - bottom.max() <= 2 * margin:
            raise ValueError("Ice thickness must exceed the interface transition widths")
        if bottom.min() <= -config.array_depth + 2 * config.grid_spacing + margin:
            raise ValueError("Acquisition stencil must remain wholly in water")
        top_limit = config.computational_top
        if config.upper_medium == "air":
            top_limit -= config.sponge_width
        if self.top + margin >= top_limit:
            raise ValueError("Ice interface enters the upper boundary/absorber")
        return self


def line_weights(coordinates):
    """Trapezoidal weights on actual nodes; a single source has unit counting weight."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) == 1:
        return np.ones(1)
    gaps = np.diff(coordinates)
    if np.any(gaps <= 0):
        raise ValueError("Acquisition coordinates must be strictly increasing")
    weights = np.empty_like(coordinates)
    weights[0], weights[-1] = gaps[0] / 2, gaps[-1] / 2
    weights[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return weights


def sampling_stencil(x, z, targets, depth, mode):
    """Return (point, stencil, [row,column]) and weights for pressure grid sampling.

    The solver injects stress with the transpose of these interpolation weights.
    Physical z decreases with row number.
    """
    targets = np.asarray(targets)
    h = x[1] - x[0]
    nodes = np.zeros((len(targets), 4, 2), dtype=np.int64)
    weights = np.zeros((len(targets), 4))
    if mode == "snap":
        nodes[:, :, 0] = np.argmin(abs(z + depth))
        nodes[:, :, 1] = np.abs(x[:, None] - targets).argmin(axis=0)[:, None]
        weights[:, 0] = 1
    else:
        fx = (targets - x[0]) / h
        fz = (z[0] + depth) / h
        ix, iz = np.floor(fx).astype(int), int(np.floor(fz))
        wx, wz = fx - ix, fz - iz
        for k, (dz, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            nodes[:, k, 0] = iz + dz
            nodes[:, k, 1] = ix + dx
            weights[:, k] = (wz if dz else 1 - wz) * (wx if dx else 1 - wx)
    if np.any(nodes[:, :, 0] < 3) or np.any(nodes[:, :, 0] >= len(z) - 3):
        raise ValueError("Vertical acquisition stencil too close to the boundary")
    if np.any(nodes[:, :, 1] < 2) or np.any(nodes[:, :, 1] >= len(x) - 2):
        raise ValueError("Horizontal acquisition stencil too close to the boundary")
    return nodes, weights
