"""Exact wet-trace condensation by retaining the last blocks of a sparse LU."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import splu


class TraceFactorError(RuntimeError):
    """The trace-restricted factorization needs the reference solve path."""


def _nested_order(indices, coordinates, degree):
    """Recursively place separators after their two subdomains on the FE grid."""
    if len(indices) <= 70:
        return indices
    locations = coordinates[:, indices]
    for axis in np.argsort(np.ptp(locations, axis=1))[::-1]:
        points = locations[axis]
        low, high = points.min(), points.max()
        split = degree * round((low + high) / (2 * degree))
        if low < split < high:
            left = points < split
            right = points > split
            return np.concatenate((
                _nested_order(indices[left], coordinates, degree),
                _nested_order(indices[right], coordinates, degree),
                indices[~(left | right)],
            ))
    return indices


class TraceFactor:
    """Cache the geometry-only permutation; factor each frequency separately."""

    def __init__(self, ice):
        self.ice = ice
        grid = np.full((ice.ndof, 2), -1, dtype=np.int32)
        z_element, r_element = np.indices(ice.element_dofs.shape[:2])
        for local in range(ice.element_dofs.shape[-1]):
            nodes = ice.element_dofs[..., local]
            z_node, r_node = divmod(local // 2, ice.degree + 1)
            ij = np.column_stack(((ice.degree * z_element + z_node).ravel(),
                                  (ice.degree * r_element + r_node).ravel()))
            valid = nodes.ravel() >= 0
            grid[nodes.ravel()[valid]] = ij[valid]
        if (grid < 0).any():
            raise ValueError("The elastic mesh contains a DOF outside element connectivity")
        self.coordinates = grid.T
        self._permutation = None
        self._trace = None

    def factor(self, s, load, average):
        """Return a numeric factor and receiver-checkable original sparse matrix."""
        trace = np.flatnonzero(load.getnnz(axis=1))
        if not trace.size or trace.size == self.ice.ndof:
            raise TraceFactorError("Need proper nonempty wetted trace and interior DOF sets")
        if self._trace is None or not np.array_equal(trace, self._trace):
            if np.any(average[:, np.setdiff1d(np.arange(self.ice.ndof), trace)].data != 0):
                raise TraceFactorError("Elastic trace averaging must be supported on loaded DOFs")
            mask = np.ones(self.ice.ndof, dtype=bool)
            mask[trace] = False
            interior = np.flatnonzero(mask)
            ordering = _nested_order(interior, self.coordinates, self.ice.degree)
            self._permutation = np.concatenate((ordering, trace))
            self._trace = trace

        matrix = self.ice.dynamic_matrix(s)
        permutation = self._permutation
        permuted_matrix = matrix[permutation, :][:, permutation].tocsc()
        try:
            numeric = splu(permuted_matrix, permc_spec="NATURAL", diag_pivot_thresh=0.01,
                           options={"SymmetricMode": True, "Equil": False})
        except RuntimeError as error:
            raise TraceFactorError(f"Trace-last sparse factorization failed: {error}") from error
        return _NumericTraceFactor(numeric, matrix, permutation, trace, load, average)


class _NumericTraceFactor:
    def __init__(self, numeric, matrix, permutation, trace, load, average):
        self.numeric = numeric
        self.matrix = matrix
        self.permutation = permutation
        self.trace = trace
        self.load = load
        self.average = average

    def panel_response(self):
        """Compute average @ A⁻¹ @ load via the trailing LU blocks."""
        numeric = self.numeric
        first_trace = self.matrix.shape[0] - len(self.trace)
        if not np.array_equal(numeric.perm_c, np.arange(self.matrix.shape[0])):
            raise TraceFactorError("Sparse LU did not preserve trace-last column ordering")
        trace_row_permutation = numeric.perm_r[first_trace:]
        if np.any(trace_row_permutation < first_trace):
            raise TraceFactorError("Sparse LU pivoting moved a trace row into the interior block")
        below = numeric.L[first_trace:, first_trace:].toarray(order="F")
        above = numeric.U[first_trace:, first_trace:].toarray(order="F")
        rhs = self.load[self.trace[np.argsort(trace_row_permutation)]].toarray(order="F")
        try:
            tmp = solve_triangular(below, rhs, lower=True, unit_diagonal=True, overwrite_b=True, check_finite=False)
            trace_solution = solve_triangular(above, tmp, overwrite_b=True, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise TraceFactorError(f"Trailing triangular solve failed: {error}") from error
        response = self.average[:, self.trace] @ trace_solution
        if not np.isfinite(response).all():
            raise TraceFactorError("Trace condensation produced nonfinite panel responses")
        return response

    def solve(self, rhs):
        """Independent full sparse backsolve in the unpermuted DOF ordering."""
        permuted = self.numeric.solve(rhs[self.permutation])
        result = np.empty_like(permuted)
        result[self.permutation] = permuted
        return result

    def relative_residual(self, solution, rhs):
        error = np.linalg.norm(self.matrix @ solution - rhs)
        norm = np.linalg.norm(rhs)
        return float(error / norm if norm else error)
