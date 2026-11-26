"""Exact element-interior condensation for tensor-product planar ice elements."""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from scipy.sparse import csr_matrix, vstack


class ElementCondensation:
    """Eliminate private element nodes while retaining every shared edge node.

    Boundary pressure loads and receivers remain on the retained degrees of
    freedom. The original displacement and residual can be recovered using the
    uncondensed element matrices.
    """

    def __init__(self, ice, *, batch_size=256):
        self.ice = ice
        self.batch_size = batch_size
        n = ice.degree + 1
        i, j = np.indices((n, n))
        is_interior = (i > 0) & (i < n - 1) & (j > 0) & (j < n - 1)
        self._interior = np.flatnonzero(np.repeat(is_interior.ravel(), 2))
        self._edge = np.flatnonzero(np.repeat(~is_interior.ravel(), 2))
        self._dofs = ice.element_dofs
        retained = np.ones(ice.ndof, dtype=bool)
        private = self._dofs[..., self._interior].ravel()
        retained[private] = False
        self.retained_dofs = np.flatnonzero(retained).astype(np.int32)
        self.ndof = len(self.retained_dofs)
        index = np.full(ice.ndof + 1, -1, dtype=np.int32)
        index[self.retained_dofs] = np.arange(self.ndof, dtype=np.int32)
        self._reduced = index[self._dofs[..., self._edge]]
        self._pml = np.zeros(ice.nr_elements, dtype=bool)
        self._pml[ice._pml_columns] = True
        self._threads = int(os.environ.get("PLANAR_ELEMENT_THREADS", "1"))
        if self._threads < 1:
            raise ValueError("PLANAR_ELEMENT_THREADS must be a positive integer")
        self._regular_blocks = tuple(self._column_blocks(False, None))
        self._indptr, self._indices, self._stripe_maps = self._assembly_pattern()
        self._upper = None

    def prepare_upper(self):
        """Cache the upper-triangular positions in the fixed CSR layout."""
        if self._upper is not None:
            return
        rows = np.repeat(np.arange(self.ndof, dtype=self._indices.dtype), np.diff(self._indptr))
        positions = np.flatnonzero(self._indices >= rows)
        row_counts = np.bincount(rows[positions], minlength=self.ndof)
        offsets = np.empty_like(self._indptr)
        offsets[0] = 0
        np.cumsum(row_counts, out=offsets[1:])
        self._upper = (offsets, self._indices[positions], positions.astype(self._indptr.dtype))

    def upper_csr(self, matrix):
        """Pack the values of a matrix returned by this object's assemble()."""
        self.prepare_upper()
        offsets, columns, positions = self._upper
        return offsets, columns, matrix.data[positions]

    def _column_blocks(self, pml, s):
        """Cache the exact quadratic row dependence of each element block.

        The ice map has row-independent thickness Jacobian. Its horizontal
        shape gradients are affine in the vertical element index, so the
        mass is invariant and the stiffness is quadratic. Interpolate it
        from three locations, including the two outermost element rows.
        """
        ice = self.ice
        last = ice.z_layers - 1
        for start in range(0, ice.nr_elements, self.batch_size):
            columns = np.arange(start, min(start + self.batch_size, ice.nr_elements))
            columns = columns[self._pml[columns] == pml]
            if not columns.size:
                continue
            rows = np.zeros(columns.shape, dtype=float)
            first, mass = ice._local_matrices(ice._element_samples(columns, rows), s)
            if not last or np.array_equal(ice._top_slope[columns], ice._bottom_slope[columns]):
                yield columns, first, mass, None, None
                continue
            middle, _ = ice._local_matrices(ice._element_samples(columns, rows + last / 2), s)
            end, _ = ice._local_matrices(ice._element_samples(columns, rows + last), s)
            linear = end - first
            quadratic = 2 * ((end - middle) - (middle - first))
            yield columns, first, mass, linear, quadratic

    def _frequency_blocks(self, s):
        blocks = (*self._regular_blocks, *self._column_blocks(True, s))
        return tuple((columns, first + s**2 * mass, linear, quadratic)
                     for columns, first, mass, linear, quadratic in blocks)

    def _element_batches(self, blocks, row):
        fraction = row / max(1, self.ice.z_layers - 1)
        for columns, first, linear, quadratic in blocks:
            if linear is None or row == 0:
                yield columns, first
            else:
                yield columns, first + fraction * (linear + (fraction - 1) * quadratic)

    def _blocks(self, matrix):
        e, i = self._edge, self._interior
        return (matrix[:, e[:, None], e], matrix[:, e[:, None], i],
                matrix[:, i[:, None], e], matrix[:, i[:, None], i])

    def _stripe_values(self, blocks, row):
        values = np.empty((self.ice.nr_elements, self._edge.size, self._edge.size),
                          dtype=np.complex128)
        for columns, matrix in self._element_batches(blocks, row):
            edge, edge_in, in_edge, diagonal = self._blocks(matrix)
            values[columns] = edge - edge_in @ np.linalg.solve(diagonal, in_edge)
        keep, local, positions = self._stripe_maps[row]
        sums = np.zeros(positions.size, dtype=np.complex128)
        np.add.at(sums, local, values[keep])
        return sums

    @staticmethod
    def _merge_stripes(row_stripes):
        stripes, pending, offset = [], None, 0
        for first, stripe in row_stripes:
            if pending is not None:
                # Adjacent element rows share only the horizontal edge nodes.
                overlap = offset + pending.shape[0] - first
                distinct = pending.shape[0] - overlap
                if distinct:
                    stripes.append(pending[:distinct])
                stripes.append(pending[distinct:] + stripe[:overlap])
                stripe = stripe[overlap:]
                first += overlap
            pending = stripe
            offset = first
        if pending is not None:
            stripes.append(pending)
        return vstack(stripes, format="csr")

    def _assembly_pattern(self):
        """Build the geometry-only CSR graph and cache element-to-entry maps."""
        stripes = []
        local_maps = []
        for dofs in self._reduced:
            rows, columns = np.broadcast_arrays(dofs[..., :, None], dofs[..., None, :])
            keep = (rows >= 0) & (columns >= 0)
            keys, local = np.unique(rows[keep].astype(np.int64) * self.ndof + columns[keep],
                                    return_inverse=True)
            row, column = np.divmod(keys, self.ndof)
            first, stop = int(row[0]), int(row[-1]) + 1
            offsets = np.r_[0, np.cumsum(np.bincount(row - first))]
            stripe = csr_matrix((np.ones(keys.size, dtype=np.int8), column, offsets),
                                shape=(stop - first, self.ndof))
            stripes.append((first, stripe))
            local_maps.append((keep, local.astype(stripe.indices.dtype), keys))
        graph = self._merge_stripes(stripes)
        graph.sort_indices()
        maps = []
        for (first, stripe), (keep, local, keys) in zip(stripes, local_maps):
            stop = first + stripe.shape[0]
            offsets = graph.indptr[first:stop + 1]
            base = offsets[0]
            rows = np.repeat(np.arange(first, stop, dtype=np.int64), np.diff(offsets))
            merged_keys = rows * self.ndof + graph.indices[base:offsets[-1]]
            positions = base + np.searchsorted(merged_keys, keys)
            maps.append((keep, local, positions.astype(graph.indptr.dtype)))
        return graph.indptr, graph.indices, maps

    def _assemble_values(self, stripe_values):
        values = np.zeros(self._indices.size, dtype=np.complex128)
        for (_, _, positions), stripe in zip(self._stripe_maps, stripe_values):
            # A position can appear in two adjacent stripes. Keep it even if they cancel exactly.
            values[positions] += stripe
        matrix = csr_matrix((values, self._indices.copy(), self._indptr.copy()),
                            shape=(self.ndof, self.ndof))
        matrix.has_canonical_format = True
        return matrix

    def assemble(self, s):
        """Assemble condensed CSR values while keeping the same pattern at every frequency."""
        blocks = self._frequency_blocks(s)
        rows = range(self.ice.z_layers)
        if self._threads == 1:
            return self._assemble_values(self._stripe_values(blocks, row) for row in rows)
        with ThreadPoolExecutor(max_workers=self._threads) as pool:
            return self._assemble_values(pool.map(lambda row: self._stripe_values(blocks, row), rows))

    def _expand(self, s, reduced, rhs, check):
        reduced = np.asarray(reduced)
        if reduced.shape != (self.ndof,):
            raise ValueError("Reduced displacement must be one vector on the retained ice nodes")
        if rhs is not None and rhs.shape != (self.ice.ndof,):
            raise ValueError("The original ice load must have one value per original degree of freedom")
        original = np.zeros(self.ice.ndof + 1, dtype=np.complex128)
        original[self.retained_dofs] = reduced
        residual = None
        if check:
            residual = np.zeros_like(original)
        e, i = self._edge, self._interior
        blocks = self._frequency_blocks(s)
        for row in range(self.ice.z_layers):
            for columns, matrix in self._element_batches(blocks, row):
                dofs = self._dofs[row, columns]
                private = dofs[:, i]
                load = -matrix[:, i[:, None], e] @ original[dofs[:, e], None]
                if rhs is not None:
                    load += rhs[private, None]
                diagonal = matrix[:, i[:, None], i]
                original[private] = np.linalg.solve(diagonal, load)[..., 0]
                if residual is not None:
                    applied = (matrix @ original[dofs, None])[..., 0]
                    np.add.at(residual, dofs.ravel(), applied.ravel())
        if residual is None:
            return original[:-1]
        error = np.linalg.norm(residual[:-1] - rhs)
        norm = np.linalg.norm(rhs)
        return original[:-1], float(error / norm if norm else error)

    def lift(self, s, reduced, rhs=None):
        """Restore each private displacement using the local interior equations.

        Any nonzero interior load must also be condensed into the retained
        right-hand side before the reduced solve.
        """
        return self._expand(s, reduced, rhs, check=False)

    def lift_and_check(self, s, reduced, rhs):
        """Restore private displacements and check the original element equations."""
        return self._expand(s, reduced, rhs, check=True)

    def relative_residual(self, s, original, rhs):
        """Evaluate the original ice equations for any supplied displacement."""
        original = np.r_[original, 0]
        residual = np.zeros_like(original, dtype=np.complex128)
        blocks = self._frequency_blocks(s)
        for row in range(self.ice.z_layers):
            for columns, matrix in self._element_batches(blocks, row):
                dofs = self._dofs[row, columns]
                applied = (matrix @ original[dofs, None])[..., 0]
                np.add.at(residual, dofs.ravel(), applied.ravel())
        error = np.linalg.norm(residual[:-1] - rhs)
        norm = np.linalg.norm(rhs)
        return float(error / norm if norm else error)
