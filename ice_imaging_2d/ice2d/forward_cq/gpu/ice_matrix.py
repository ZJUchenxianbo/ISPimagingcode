"""Fixed sparse layout for the GPU elastic assembly and original residual."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


@dataclass(frozen=True)
class _SparseLayout:
    indices: np.ndarray
    offsets: np.ndarray


def _positions(offsets):
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))


def _packed(row, column, size):
    return column.astype(np.int64) * size + row


class GpuIceMatrix:
    """Cache a numerical CSC and upper CSR structure without retaining dense PML tensors.

    The CPU elastic model still computes PML element matrices with its original
    quadrature. The layout preserves potential structural zeros so cuDSS can
    reuse its symbolic analysis across all Laplace samples.
    """

    def __init__(self, ice):
        self.ice = ice
        n = ice.ndof
        self.shape = (n, n)
        stiffness, mass = ice._k, ice._m
        if ice._pml_columns.size:
            pml_rows, pml_columns = ice._pml_coordinates
            pml = coo_matrix((np.ones(pml_rows.size, dtype=bool), (pml_rows, pml_columns)), shape=self.shape).tocsc()
        else:
            pml_rows = pml_columns = np.empty(0, dtype=np.int32)
            pml = csc_matrix(self.shape, dtype=bool)
        pattern = stiffness.astype(bool).tocsc() + mass.astype(bool).tocsc() + pml
        pattern.sort_indices()
        if max(n, pattern.nnz) >= np.iinfo(np.int32).max:
            raise ValueError("The GPU elastic layout uses 32-bit sparse indices")
        self._full = _SparseLayout(pattern.indices.astype(np.int32, copy=False), pattern.indptr.astype(np.int32, copy=False))
        full_column = _positions(pattern.indptr)
        addresses = _packed(pattern.indices, full_column, n)

        stiffness_active = stiffness.data != 0
        stiffness_addresses = _packed(_positions(stiffness.indptr)[stiffness_active],
                                      stiffness.indices[stiffness_active], n)
        self._stiffness = np.zeros(pattern.nnz, dtype=stiffness.dtype)
        self._stiffness[np.searchsorted(addresses, stiffness_addresses)] = stiffness.data[stiffness_active]
        del stiffness_active, stiffness_addresses

        # Sparse mass has only two diagonal component blocks. Keep a compact
        # scatter map and reference its immutable values in the CPU ice model.
        mass_active = mass.data != 0
        mass_addresses = _packed(_positions(mass.indptr)[mass_active], mass.indices[mass_active], n)
        self._mass_positions = np.searchsorted(addresses, mass_addresses).astype(np.int32)
        self._mass_data = mass.data if mass_active.all() else mass.data[mass_active]
        del mass_active, mass_addresses

        pml_positions = np.searchsorted(addresses, _packed(pml_rows, pml_columns, n))
        self._pml_order = np.argsort(pml_positions, kind="stable").astype(np.int32)
        self._pml_positions, pml_counts = np.unique(pml_positions, return_counts=True)
        self._pml_positions = self._pml_positions.astype(np.int32, copy=False)
        self._pml_starts = np.r_[0, np.cumsum(pml_counts, dtype=np.int32)[:-1]] if pml_counts.size else pml_counts
        del addresses, pml_positions

        # Sort physical upper-triangle entries by row for cuDSS. Use the actual
        # upper structure because roundoff can make stored zeros asymmetric.
        upper = pattern.indices <= full_column
        upper_address = _packed(full_column[upper], pattern.indices[upper], n)
        order = np.argsort(upper_address, kind="stable")
        self._upper_positions = np.flatnonzero(upper)[order].astype(np.int32)
        upper_address = upper_address[order]
        upper_count = np.bincount(upper_address // n, minlength=n)
        upper_offsets = np.empty(n + 1, dtype=np.int32)
        upper_offsets[0] = 0
        np.cumsum(upper_count, dtype=np.int32, out=upper_offsets[1:])
        self._upper = _SparseLayout((upper_address % n).astype(np.int32), upper_offsets)

    def cache_bytes(self):
        """Retained data added by the layout beyond the original ice model."""
        arrays = (self._stiffness, self._mass_positions, self._pml_positions, self._pml_starts,
                  self._pml_order, self._upper_positions, self._full.indices, self._full.offsets,
                  self._upper.indices, self._upper.offsets)
        additional_mass = 0 if self._mass_data is self.ice._m.data else self._mass_data.nbytes
        return sum(array.nbytes for array in arrays) + additional_mass

    def assemble(self, s):
        """Return (original-layout CSC, (CSR offsets, column indices, values))."""
        data = self._stiffness.astype(np.complex128)
        data[self._mass_positions] += s**2 * self._mass_data
        if self._pml_positions.size:
            ice = self.ice
            stiffness, mass = ice._local_matrices(ice._pml_samples, s)
            values = np.broadcast_to(stiffness + s**2 * mass, ice._pml_keep.shape)[ice._pml_keep]
            data[self._pml_positions] += np.add.reduceat(values[self._pml_order], self._pml_starts)
        matrix = csc_matrix((data, self._full.indices, self._full.offsets), shape=self.shape)
        return matrix, (self._upper.offsets, self._upper.indices, data[self._upper_positions])
