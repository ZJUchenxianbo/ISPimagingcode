"""Cartesian plane-strain ice FEM with independently shaped top and bottom."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu

from .basis import lagrange_basis


@dataclass(frozen=True)
class ElementSamples:
    x: np.ndarray
    values: np.ndarray
    dx: np.ndarray
    dz: np.ndarray
    weights: np.ndarray


class PlanarIce:
    """Use Q elements between two piecewise-linear interfaces sampled at edges."""

    def __init__(self, edges, bottom, top, *, stretch=None, x_bounds=None, z_layers=18,
                 degree=4, panels_per_element=3, element_size=None, clamp_outer=True,
                 rho=900.0, cp=3600.0, cs=1800.0, build_matrices=True, build_global_matrices=True):
        self.edges = np.asarray(edges, dtype=float)
        self.bottom = np.asarray(bottom, dtype=float)
        self.top = np.asarray(top, dtype=float)
        if (self.edges.ndim != 1 or self.edges.size < 2 or not np.isfinite(self.edges).all()
                or np.any(np.diff(self.edges) <= 0) or self.bottom.shape != self.edges.shape
                or self.top.shape != self.edges.shape or not np.isfinite([self.bottom, self.top]).all()
                or (self.top <= self.bottom).any()):
            raise ValueError("Ice geometry requires increasing edges and a top strictly above the bottom")
        if degree < 1 or z_layers < 1 or panels_per_element < 1 or rho <= 0 or cs <= 0 or cp <= cs:
            raise ValueError("Invalid planar ice mesh or material parameters")
        if element_size is not None and (not np.isfinite(element_size) or element_size <= 0):
            raise ValueError("The target ice element size must be positive and finite")
        self.n_panels = self.edges.size - 1
        self.degree = degree
        self.z_layers = z_layers
        self.stretch = stretch
        self.rho = rho
        self.mu = rho * cs**2
        self.lam = rho * cp**2 - 2.0 * self.mu
        self._traces = {}

        joins = np.array([0, self.n_panels])
        if stretch is not None:
            if x_bounds is None:
                raise ValueError("The physical lateral bounds are required with horizontal PML")
            start, stop = x_bounds
            if not self.edges[0] <= start < stop <= self.edges[-1]:
                raise ValueError("The lateral PML must be outside the physical ice domain")
            bounds = [int(np.argmin(abs(self.edges - bound))) for bound in x_bounds]
            if not np.allclose(self.edges[bounds], x_bounds, atol=1e-10, rtol=0):
                raise ValueError("Both PML entry points must be mesh edges")
            joins = np.unique(np.r_[0, bounds, self.n_panels])
        if element_size is None:
            cuts = np.unique(np.r_[np.arange(0, self.n_panels, panels_per_element), joins])
        else:
            # Keep the BEM trace panels and PML joins intact. The FEM shape
            # functions span multiple panels, with quadrature split at each one.
            cuts = [0]
            for stop in joins[1:]:
                while cuts[-1] < stop:
                    start = cuts[-1]
                    next_cut = np.searchsorted(self.edges, self.edges[start] + element_size, side="right") - 1
                    cuts.append(min(stop, max(start + 1, next_cut)))
            cuts = np.array(cuts, dtype=int)

        self.cuts = cuts
        self.element_edges = self.edges[cuts]
        self.nr_elements = cuts.size - 1
        nx = degree * self.nr_elements + 1
        nz = degree * z_layers + 1
        active = np.ones((nz, nx, 2), dtype=bool)
        if clamp_outer:
            active[:, [0, -1], :] = False
        self.ndof = int(active.sum())
        if not self.ndof:
            raise ValueError("No unconstrained ice degrees of freedom")
        if not build_matrices:
            return
        node_dofs = np.full(active.shape, -1, dtype=np.int32)
        node_dofs[active] = np.arange(self.ndof, dtype=np.int32)
        local = np.arange(degree + 1)
        j = degree * np.arange(z_layers)[:, None, None, None] + local[None, None, :, None]
        i = degree * np.arange(self.nr_elements)[None, :, None, None] + local[None, None, None, :]
        self.element_dofs = node_dofs[j, i].reshape(z_layers, self.nr_elements, -1)
        self._interface_dofs = {"bottom": node_dofs[0], "top": node_dofs[-1]}
        self._set_quadrature()
        pml = np.zeros(self.nr_elements, dtype=bool)
        if stretch is not None:
            pml = (cuts[:-1] < bounds[0]) | (cuts[1:] > bounds[1])
        self._pml_columns = np.flatnonzero(pml)

        if not build_global_matrices:
            return
        self._k, self._m = self._assemble_regular(~pml)
        if self._pml_columns.size:
            dofs = self.element_dofs[:, self._pml_columns]
            self._pml_coordinates, self._pml_keep = self._assembly_coordinates(dofs)
            local_slope = self._top_slope[self._pml_columns] - self._bottom_slope[self._pml_columns]
            if np.all(local_slope == 0.0):
                self._pml_samples = self._element_samples(self._pml_columns, np.zeros_like(self._pml_columns))
            else:
                j, i = np.indices(dofs.shape[:2])
                self._pml_samples = self._element_samples(self._pml_columns[i], j)

    def _set_quadrature(self):
        # Split quadrature at every change of interface slope within a Q cell.
        points, weights = leggauss(self.degree + 1)
        panels = self.cuts[:-1, None] + np.arange(np.diff(self.cuts).max())[None, :]
        active = panels < self.cuts[1:, None]
        panels = np.minimum(panels, self.cuts[1:, None] - 1)
        left, right = self.edges[panels], self.edges[panels + 1]
        panel_width = right - left
        x = left[..., None] + 0.5 * panel_width[..., None] * (points + 1.0)
        elem_width = np.diff(self.element_edges)
        xi = 2.0 * (x - self.element_edges[:-1, None, None]) / elem_width[:, None, None] - 1.0
        h, dh = lagrange_basis(self.degree, xi.ravel())
        h = h.reshape(self.nr_elements, -1, points.size)
        dh = dh.reshape(h.shape) * (2.0 / elem_width[:, None, None])
        v, dv = lagrange_basis(self.degree, points)
        self._values = np.einsum("ja,iqb->ijqab", v, h).reshape(self.nr_elements, -1, points.size**2)
        self._dx_ref = np.einsum("ja,iqb->ijqab", v, dh).reshape(self._values.shape)
        self._dz_ref = np.einsum("ja,iqb->ijqab", dv, h).reshape(self._values.shape)
        self._eta = np.repeat(points, x.shape[1] * x.shape[2])

        def repeat_vertical(array):
            return np.tile(array.reshape(self.nr_elements, -1), (1, points.size))

        height = self.top - self.bottom
        t = 0.5 * (points + 1.0)
        thickness = height[panels, None] * (1.0 - t) + height[panels + 1, None] * t
        self._thickness = repeat_vertical(thickness)
        self._x = repeat_vertical(x)
        self._bottom_slope = repeat_vertical(
            (np.diff(self.bottom)[panels] / panel_width)[..., None] * np.ones_like(points)
        )
        self._top_slope = repeat_vertical(
            (np.diff(self.top)[panels] / panel_width)[..., None] * np.ones_like(points)
        )
        scaled_weights = 0.5 * panel_width[..., None] * active[..., None] * weights
        self._weights = np.einsum("j,ipq->ijpq", weights, scaled_weights).reshape(self.nr_elements, -1)

    def _element_samples(self, i, j):
        fraction = (j[..., None] + 0.5 * (1.0 + self._eta)) / self.z_layers
        height_slope = (1.0 - fraction) * self._bottom_slope[i] + fraction * self._top_slope[i]
        dz_deta = self._thickness[i] / (2.0 * self.z_layers)
        dz = self._dz_ref[i] / dz_deta[..., None]
        dx = self._dx_ref[i] - dz * height_slope[..., None]
        return ElementSamples(self._x[i], self._values[i], dx, dz, self._weights[i] * dz_deta)

    def _local_matrices(self, samples, s=None):
        dx, dz, jacobian = samples.dx, samples.dz, samples.weights
        if s is not None:
            _, derivative = self.stretch.horizontal(samples.x, s)
            dx = dx / derivative[..., None]
            jacobian = jacobian * derivative

        def bilinear(a, b):
            return a.swapaxes(-1, -2) @ (jacobian[..., None] * b)

        xx, zz = bilinear(dx, dx), bilinear(dz, dz)
        xz = bilinear(dx, dz)
        count = dx.shape[-1]
        shape = dx.shape[:-2] + (2 * count, 2 * count)
        stiffness = np.empty(shape, dtype=xx.dtype)
        stiffness[..., 0::2, 0::2] = (self.lam + 2.0 * self.mu) * xx + self.mu * zz
        stiffness[..., 1::2, 1::2] = (self.lam + 2.0 * self.mu) * zz + self.mu * xx
        offdiag = self.lam * xz + self.mu * xz.swapaxes(-1, -2)
        stiffness[..., 0::2, 1::2] = offdiag
        stiffness[..., 1::2, 0::2] = offdiag.swapaxes(-1, -2)
        scalar_mass = self.rho * bilinear(samples.values, samples.values)
        mass = np.zeros_like(stiffness)
        mass[..., 0::2, 0::2] = scalar_mass
        mass[..., 1::2, 1::2] = scalar_mass
        return stiffness, mass

    @staticmethod
    def _assembly_coordinates(dofs):
        rows, cols = np.broadcast_arrays(dofs[..., :, None], dofs[..., None, :])
        keep = (rows >= 0) & (cols >= 0)
        return (rows[keep], cols[keep]), keep

    def _assemble_regular(self, regular):
        mask = np.broadcast_to(regular, self.element_dofs.shape[:2])
        z, x = np.nonzero(mask)
        shape = (self.ndof, self.ndof)
        levels = {}
        for start in range(0, x.size, 128):
            segment = slice(start, start + 128)
            i, j = x[segment], z[segment]
            coordinates, keep = self._assembly_coordinates(self.element_dofs[j, i])
            block = tuple(coo_matrix((values[keep], coordinates), shape=shape).tocsr()
                          for values in self._local_matrices(self._element_samples(i, j)))
            level = 0
            while level in levels:
                block = tuple(a + b for a, b in zip(levels.pop(level), block))
                level += 1
            levels[level] = block
        stiffness, mass = sum_blocks(levels.values(), shape)
        stiffness.eliminate_zeros()
        mass.eliminate_zeros()
        return stiffness, mass

    def dynamic_matrix(self, s):
        """Return plane-strain K + s² M, including the frequency-dependent horizontal PML."""
        matrix = self._k + s**2 * self._m
        if self._pml_columns.size:
            stiffness, mass = self._local_matrices(self._pml_samples, s)
            local = np.broadcast_to(stiffness + s**2 * mass, self._pml_keep.shape)
            matrix += coo_matrix((local[self._pml_keep], self._pml_coordinates), shape=matrix.shape).tocsr()
        return matrix.tocsc()

    def factor(self, s):
        """Factor one complex-frequency elastic matrix."""
        return splu(self.dynamic_matrix(s))

    def surface_displacement_operator(self, x):
        """Sample both elastic surfaces at x, with top before bottom and x before z."""
        x = np.asarray(x, dtype=float)
        if (x.ndim != 1 or not np.isfinite(x).all()
                or np.any((x < self.edges[0]) | (x > self.edges[-1]))):
            raise ValueError("Displacement sample points must lie on the ice mesh")
        element = np.minimum(np.searchsorted(self.element_edges, x, side="right") - 1, self.nr_elements - 1)
        left, right = self.element_edges[element], self.element_edges[element + 1]
        basis, _ = lagrange_basis(self.degree, 2 * (x - left) / (right - left) - 1)
        nodes = element[:, None] * self.degree + np.arange(self.degree + 1)
        dofs = np.stack([self._interface_dofs[side][nodes] for side in ("top", "bottom")])
        rows = np.broadcast_to(np.arange(4 * len(x)).reshape(2, len(x), 1, 2), dofs.shape)
        values = np.broadcast_to(basis[None, :, :, None], dofs.shape)
        keep = dofs >= 0
        return csr_matrix((values[keep], (rows[keep], dofs[keep])), shape=(4 * len(x), self.ndof))

    def _setup_trace(self, side, panel_edges):
        edges = self.edges if panel_edges is None else np.asarray(panel_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("Invalid edges for an ice boundary trace")
        cache = self._traces.get(side)
        if cache is not None and np.array_equal(cache[0], edges):
            return cache[1:]

        left, right = edges[:-1], edges[1:]
        center = 0.5 * (left + right)
        original = np.searchsorted(self.edges, center, side="right") - 1
        if (np.any(original < 0) or np.any(original >= self.n_panels)
                or np.any(left < self.edges[original] - 1e-10)
                or np.any(right > self.edges[original + 1] + 1e-10)):
            raise ValueError("Trace panels must subdivide the geometry panels")
        geometry = self.bottom if side == "bottom" else self.top
        slope = np.diff(geometry)[original] / np.diff(self.edges)[original]
        elem = np.searchsorted(self.cuts, original, side="right") - 1
        elem_left, elem_right = self.element_edges[elem], self.element_edges[elem + 1]
        points, weight = leggauss(self.degree + 2)
        width = right - left
        x = left[:, None] + width[:, None] * (1.0 + points) / 2.0
        xi = 2.0 * (x - elem_left[:, None]) / (elem_right - elem_left)[:, None] - 1.0
        basis, _ = lagrange_basis(self.degree, xi.ravel())
        basis = basis.reshape(x.shape + (self.degree + 1,))
        dofs = self._interface_dofs[side][elem[:, None] * self.degree + np.arange(self.degree + 1)]
        panels = np.broadcast_to(np.arange(width.size)[:, None, None], dofs.shape)
        keep = dofs >= 0
        data = (x, slope[:, None], weight * width[:, None] / 2.0, basis, (dofs[keep], panels[keep]), keep)
        self._traces[side] = (edges.copy(), *data)
        return data

    def coupling(self, s, side="bottom", panel_edges=None):
        """Return maps p→−p n (traction), displacement→average u·n, with n outward of ice."""
        if side not in self._interface_dofs:
            raise ValueError("The ice trace must be 'bottom' or 'top'")
        x, slope, weight, basis, coordinates, keep = self._setup_trace(side, panel_edges)
        dx = np.ones_like(x) if self.stretch is None else self.stretch.horizontal(x, s)[1]
        length = np.sum(weight * np.sqrt(dx**2 + slope**2), axis=1)
        normal_measure = np.stack((np.broadcast_to(-slope, x.shape), dx), axis=-1)
        sign = 1 if side == "top" else -1
        normal_load = np.einsum("pq,pqi,pqd->pid", weight, basis, normal_measure) * sign
        load = coo_matrix((-normal_load[keep], coordinates), shape=(self.ndof, x.shape[0])).tocsr()
        load.eliminate_zeros()
        return load, (-diags(1.0 / length) @ load.T).tocsr()


def sum_blocks(blocks, shape):
    """Reduce the remaining balanced sparse assembly blocks."""
    stiffness, mass = csr_matrix(shape), csr_matrix(shape)
    for k, m in blocks:
        stiffness += k
        mass += m
    return stiffness, mass
