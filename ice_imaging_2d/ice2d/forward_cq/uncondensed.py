"""Fixed upper CSR assembly of the original planar ice FEM equations."""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .condensed import ElementCondensation


@lru_cache(maxsize=1)
def _native_library():
    source = Path(__file__).with_suffix(".cpp")
    output = source.with_name("_planar_uncondensed.so")
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if not source.is_file():
        return None
    try:
        if not output.exists() or output.stat().st_mtime_ns < source.stat().st_mtime_ns:
            try:
                subprocess.run([
                    "c++", "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                    str(source), "-o", str(temporary),
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        lib = ctypes.CDLL(str(output))
    except (OSError, subprocess.CalledProcessError):
        return None
    count, ptr, real = ctypes.c_size_t, ctypes.c_void_p, ctypes.c_double
    lib.planar_ice_physical.argtypes = (count, ptr, ptr, real, real, ptr)
    lib.planar_ice_physical.restype = None
    lib.planar_ice_pml.argtypes = (count, ptr, ptr, ptr, ptr, ptr, real, ptr)
    lib.planar_ice_pml.restype = None
    return lib


def _upper_coordinates(dofs):
    rows, columns = np.broadcast_arrays(dofs[..., :, None], dofs[..., None, :])
    keep = (rows >= 0) & (columns >= rows)
    return rows[keep], columns[keep], keep


@dataclass(frozen=True, eq=False)
class IceAssemblyLayout:
    """Shape-independent connectivity and scatter indices for one ice mesh topology."""

    ndof: int
    degree: int
    batch_size: int
    element_dofs: np.ndarray
    pml_columns: np.ndarray
    offsets: np.ndarray
    columns: np.ndarray
    physical_maps: tuple
    pml_maps: tuple
    device_pml_maps: tuple

    def matches(self, ice, batch_size):
        return (self.ndof == ice.ndof and self.degree == ice.degree
                and self.batch_size == batch_size
                and np.array_equal(self.element_dofs, ice.element_dofs)
                and np.array_equal(self.pml_columns, ice._pml_columns))


def _readonly(indices):
    indices.flags.writeable = False
    return indices


class PlanarUncondensed:
    """Cache the physical part of the full, complex-symmetric ice operator.

    The upper CSR graph includes the PML and is independent of frequency. Only
    the mass term and the stretched PML entries change at a new Laplace point.
    """

    def __init__(self, ice, *, batch_size=128, native=True, layout=None, cache_layout=False):
        self.ice = ice
        self.ndof = ice.ndof
        self.batch_size = batch_size
        if layout is not None and not layout.matches(ice, batch_size):
            raise ValueError("The cached ice assembly layout does not match the ice connectivity or PML columns")
        self.assembly_layout = layout
        self._native = _native_library() if native else None
        pml = np.zeros(ice.nr_elements, dtype=bool)
        pml[ice._pml_columns] = True
        self._physical = np.flatnonzero(~pml)
        self._pml = np.flatnonzero(pml)
        self._pml_degree = ice.z_layers - 1 if np.any(
            ice._top_slope[self._pml] != ice._bottom_slope[self._pml]
        ) else 0
        if self._pml.size:
            rows = (0, self._pml_degree / 2, self._pml_degree) if self._pml_degree else (0,)
            self._pml_samples = tuple(ice._element_samples(self._pml, np.full(self._pml.shape, row, dtype=float))
                                      for row in rows)
        else:
            self._pml_samples = ()
        self._device_layout = None

        if layout is None:
            graph = ElementCondensation._merge_stripes(self._stripe(dofs) for dofs in ice.element_dofs)
            graph.sort_indices()
            if max(self.ndof, graph.nnz) >= np.iinfo(np.int32).max:
                raise ValueError("The uncondensed GPU ice layout uses 32-bit CSR indices")
            self._offsets = _readonly(graph.indptr)
            self._columns = _readonly(graph.indices)
        else:
            self._offsets, self._columns = layout.offsets, layout.columns
        self._stiffness = np.zeros(self._columns.size, dtype=np.float64)
        mass = np.zeros_like(self._stiffness)
        if layout is None:
            physical_maps = self._assemble_physical(mass, cache_layout)
            if cache_layout:
                self.assembly_layout = IceAssemblyLayout(
                    ice.ndof, ice.degree, batch_size, _readonly(ice.element_dofs.copy()),
                    _readonly(ice._pml_columns.copy()), self._offsets, self._columns,
                    physical_maps, self._pml_maps, self._group_pml_maps())
        else:
            self._pml_maps = layout.pml_maps
            batches = tuple(self._batches(self._physical))
            for row, row_maps in enumerate(layout.physical_maps):
                for columns, (sources, positions) in zip(batches, row_maps):
                    stiff, inertial = self._element_matrices(columns, row)
                    np.add.at(self._stiffness, positions, stiff.ravel()[sources])
                    np.add.at(mass, positions, inertial.ravel()[sources])

        if self._native:
            self._mass = mass
        else:
            self._mass_positions = np.flatnonzero(mass).astype(np.int32)
            self._mass = mass[self._mass_positions]

    def _assemble_physical(self, mass, cache_layout):
        physical_maps, pml_maps = [], []
        for row, dofs in enumerate(self.ice.element_dofs):
            start_row, stop_row = int(dofs[dofs >= 0].min()), int(dofs.max()) + 1
            pointers = self._offsets[start_row:stop_row + 1]
            rows = np.repeat(np.arange(start_row, stop_row, dtype=np.int64), np.diff(pointers))
            first = pointers[0]
            keys = rows * self.ndof + self._columns[first:pointers[-1]]
            row_maps = []
            for columns in self._batches(self._physical):
                keep, positions = self._scatter_map(dofs[columns], first, keys)
                stiff, inertial = self._element_matrices(columns, row)
                np.add.at(self._stiffness, positions, stiff[keep])
                np.add.at(mass, positions, inertial[keep])
                if cache_layout:
                    row_maps.append(self._compact_map(keep, positions))
            if cache_layout:
                physical_maps.append(tuple(row_maps))
            if self._pml.size:
                keep, positions = self._scatter_map(dofs[self._pml], first, keys)
                pml_maps.append(self._compact_map(keep, positions))
        self._pml_maps = tuple(pml_maps)
        return tuple(physical_maps)

    @staticmethod
    def _compact_map(keep, positions):
        sources = np.flatnonzero(keep).astype(np.int32)
        return _readonly(sources), _readonly(positions.astype(np.int32))

    @property
    def uses_native_assembly(self):
        return self._native is not None

    def _batches(self, columns):
        for start in range(0, len(columns), self.batch_size):
            yield columns[start:start + self.batch_size]

    def _stripe(self, dofs):
        rows, columns, _ = _upper_coordinates(dofs)
        first, stop = int(rows.min()), int(rows.max()) + 1
        # At most four elements meet at one vertex, so the summed markers fit int8.
        stripe = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows - first, columns)),
                            shape=(stop - first, self.ndof)).tocsr()
        return first, stripe

    def _scatter_map(self, dofs, first, keys):
        rows, columns, keep = _upper_coordinates(dofs)
        positions = first + np.searchsorted(keys, rows.astype(np.int64) * self.ndof + columns)
        return keep, positions

    def _element_matrices(self, columns, row, s=None):
        rows = np.full(columns.shape, row, dtype=float)
        return self.ice._local_matrices(self.ice._element_samples(columns, rows), s)

    def pml_coefficients(self, s):
        """Return the per-frequency polynomial representation of the PML element matrices."""
        if not self._pml_samples:
            return None, None, None
        first, mass = self.ice._local_matrices(self._pml_samples[0], s)
        constant = first + s**2 * mass
        linear = quadratic = None
        if self._pml_degree:
            middle, _ = self.ice._local_matrices(self._pml_samples[1], s)
            end, _ = self.ice._local_matrices(self._pml_samples[2], s)
            # The shape gradients are affine in the row index. Stiffness is quadratic.
            linear = end - first
            quadratic = 2 * ((end - middle) - (middle - first))
        return constant, linear, quadratic

    @property
    def upper_pattern(self):
        """Immutable row offsets and column indices of the elastic upper CSR."""
        return self._offsets, self._columns

    def prepare_device_assembly(self):
        """Group PML contributions per entry, retaining the original summation order."""
        if self._device_layout is not None:
            return self._device_layout
        mass = self._mass
        if not self._native:
            mass = np.zeros_like(self._stiffness)
            mass[self._mass_positions] = self._mass
        count = self._pml.size * self.ice.element_dofs.shape[-1]**2
        levels = 3 if self._pml_degree else 1
        pml_maps = (self.assembly_layout.device_pml_maps if self.assembly_layout is not None
                    else self._group_pml_maps())
        self._device_layout = (self._stiffness, mass, *pml_maps, self._pml_degree, count, levels)
        return self._device_layout

    def _group_pml_maps(self):
        if self._pml_maps:
            targets = np.concatenate([positions for _, positions in self._pml_maps])
            order = np.argsort(targets, kind="stable")
            targets, lengths = np.unique(targets[order], return_counts=True)
            offsets = np.empty(targets.size + 1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(lengths, dtype=np.int32, out=offsets[1:])
            sources = np.concatenate([source for source, _ in self._pml_maps])[order]
            rows = np.repeat(np.arange(self.ice.z_layers, dtype=np.int32),
                             [len(positions) for _, positions in self._pml_maps])[order]
        else:
            targets = sources = rows = np.empty(0, dtype=np.int32)
            offsets = np.zeros(1, dtype=np.int32)
        return tuple(_readonly(indices) for indices in (targets, offsets, sources, rows))

    def _add_pml(self, values, s):
        if not self._pml.size:
            return
        constant, linear, quadratic = self.pml_coefficients(s)
        last = self.ice.z_layers - 1
        if self._native:
            linear_ptr = linear.ctypes.data if linear is not None else None
            quadratic_ptr = quadratic.ctypes.data if quadratic is not None else None
            for row, (sources, positions) in enumerate(self._pml_maps):
                self._native.planar_ice_pml(sources.size, sources.ctypes.data, positions.ctypes.data,
                                            constant.ctypes.data, linear_ptr, quadratic_ptr,
                                            row / max(last, 1), values.ctypes.data)
            return
        for row, (sources, positions) in enumerate(self._pml_maps):
            blocks = constant
            if linear is not None and row:
                z = row / last
                blocks = constant + z * (linear + (z - 1) * quadratic)
            np.add.at(values, positions, blocks.ravel()[sources])

    def assemble(self, s):
        """Return the upper half of the original complex-symmetric ice matrix."""
        if self._native:
            values = np.empty(self._stiffness.size, dtype=np.complex128)
            wave = s**2
            self._native.planar_ice_physical(values.size, self._stiffness.ctypes.data, self._mass.ctypes.data,
                                             wave.real, wave.imag, values.ctypes.data)
        else:
            values = self._stiffness.astype(np.complex128)
            values[self._mass_positions] += s**2 * self._mass
        self._add_pml(values, s)
        matrix = csr_matrix((values, self._columns, self._offsets), shape=(self.ndof, self.ndof))
        matrix.has_canonical_format = True
        return matrix

    @staticmethod
    def upper_csr(matrix):
        """Return the cuDSS buffers for a matrix produced by assemble()."""
        return matrix.indptr, matrix.indices, matrix.data

    @staticmethod
    def relative_residual(matrix, solution, rhs):
        """Evaluate the full original equation from the stored upper triangle."""
        error = matrix @ solution + matrix.T @ solution - matrix.diagonal() * solution - rhs
        norm = np.linalg.norm(rhs)
        return float(np.linalg.norm(error) / norm if norm else np.linalg.norm(error))
