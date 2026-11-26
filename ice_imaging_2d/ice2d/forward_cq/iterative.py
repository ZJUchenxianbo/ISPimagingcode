"""Solve the water/ice interface system without constructing its ice response."""

from dataclasses import dataclass
from contextlib import nullcontext
from time import perf_counter

import numpy as np
from scipy.fft import fft, fftfreq, ifft, next_fast_len
from scipy.linalg import get_lapack_funcs, lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class WaterSolution:
    solution: np.ndarray
    displacement: np.ndarray
    normal: np.ndarray
    relative_residual: float
    iterations: int
    preconditioner_s: float
    elapsed: float


class WaterConvergenceError(RuntimeError):
    """The original interface equations did not reach the requested tolerance."""


def _local_compliance(s, ice_density, ice_vp, ice_thickness):
    """Pressure to ice outward-normal displacement, for preconditioning only."""
    compliance = -1.0 / (ice_density * ice_vp * s)
    if ice_thickness is None:
        return compliance
    return compliance / np.tanh(s * np.asarray(ice_thickness) / ice_vp)


def _linear_interpolation(source, targets):
    left = np.searchsorted(source, targets, side="right") - 1
    left = left.clip(0, source.size - 2)
    right = left + 1
    weights = ((targets - source[left]) / (source[right] - source[left])).clip(0.0, 1.0)
    indices = np.column_stack((left, right)).ravel()
    values = np.column_stack((1.0 - weights, weights)).ravel()
    offsets = 2 * np.arange(targets.size + 1)
    return csr_matrix((values, indices, offsets), shape=(targets.size, source.size))


def _spectral_preconditioner(water, coupling, s, x, *, thickness, density, vp, vs):
    """Use flat-ice P–SV compliance in the acoustic preconditioner."""
    from .air import flat_compliance

    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    if x.shape != (coupling.shape[1],) or not dx.size or not np.isfinite(x).all() or np.any(dx <= 0.0):
        raise ValueError("The spectral preconditioner requires increasing wet collocation coordinates")
    thickness = np.asarray(thickness, dtype=float)
    if (thickness.shape not in ((), x.shape) or not np.isfinite(thickness).all()
            or np.any(thickness <= 0.0)):
        raise ValueError("The spectral preconditioner requires positive ice thickness, scalar or per wet panel")
    # The auxiliary grid is used only by the preconditioner. Twice the
    # physical grid size leaves room to pad the linear convolution.
    count = int(np.ceil((x[-1] - x[0]) / np.min(dx))) + 1
    uniform = np.linspace(x[0], x[-1], count)
    fft_count = next_fast_len(2 * count)
    wave_number = 2.0 * np.pi * fftfreq(fft_count, d=uniform[1] - uniform[0])
    to_uniform = _linear_interpolation(x, uniform)
    from_uniform = _linear_interpolation(uniform, x)
    smallest, largest = float(thickness.min()), float(thickness.max())
    if largest == smallest:
        references = ((smallest, from_uniform),)
    else:
        # Blend the displacement at each wet panel between neighbouring
        # constant-thickness slabs. Only the preconditioner uses this model.
        middle = 0.5 * (smallest + largest)
        heights = np.array([smallest, middle, largest])
        weights = _linear_interpolation(heights, thickness).T.tocsr()
        references = tuple((height, from_uniform.multiply(weights[j].toarray().T).tocsr())
                           for j, height in enumerate(heights))
    symbols = [(-flat_compliance(s, wave_number, height, ice_density=density, ice_vp=vp,
                                 ice_vs=vs, air_loading=False), mapping)
               for height, mapping in references]
    approx = np.array(water, order="F", copy=True)
    for first in range(0, water.shape[0], 64):
        rows = slice(first, first + 64)
        for symbol, mapping in symbols:
            regular = coupling[rows] @ mapping
            regular = fft(regular, n=fft_count, axis=1)
            regular *= symbol
            regular = ifft(regular, axis=1, overwrite_x=True)
            approx[rows, :x.size] -= regular[:, :count] @ to_uniform
    return approx


def _preconditioner(water, coupling, method, bandwidth, s, *,
                    ice_density, ice_vp, ice_vs, ice_thickness, pressure_x, cuda):
    size, trace_size = coupling.shape
    if method == "none":
        return None
    if not isinstance(water, np.ndarray) or not isinstance(coupling, np.ndarray):
        raise ValueError("This preconditioner requires the assembled water boundary blocks")
    if method in ("dense", "banded"):
        compliance = np.broadcast_to(_local_compliance(s, ice_density, ice_vp, ice_thickness), trace_size)
    if method in ("dense", "spectral"):
        if method == "spectral":
            approx = _spectral_preconditioner(water, coupling, s, pressure_x, thickness=ice_thickness,
                                             density=ice_density, vp=ice_vp, vs=ice_vs)
        else:
            # No dense ice response or array of extra right-hand sides is needed.
            approx = np.array(water, order="F", copy=True)
            for first in range(0, trace_size, 32):
                columns = slice(first, min(first + 32, trace_size))
                approx[:, columns] -= coupling[:, columns] * compliance[columns]
        if cuda is not None:
            return cuda.factor(approx)
        factor = lu_factor(approx, overwrite_a=True, check_finite=False)

        def solve(vector):
            return lu_solve(factor, vector, check_finite=False)

    elif method == "banded":
        # LAPACK's general band storage has an extra lower-bandwidth workspace.
        # Discarding far entries affects only this preconditioner.
        bandwidth = min(bandwidth, size - 1)
        band = np.zeros((3 * bandwidth + 1, size), dtype=np.complex128, order="F")
        for offset in range(-bandwidth, bandwidth + 1):
            columns = np.arange(max(0, -offset), min(size, size - offset))
            rows = columns + offset
            values = water[rows, columns]
            wet = columns < trace_size
            values[wet] -= coupling[rows[wet], columns[wet]] * compliance[columns[wet]]
            band[2 * bandwidth + offset, columns] = values
        factor_band, solve_band = get_lapack_funcs(("gbtrf", "gbtrs"), (band,))
        factor, pivots, status = factor_band(band, bandwidth, bandwidth, overwrite_ab=True)
        if status:
            raise ValueError(f"Banded water preconditioner factorization failed (LAPACK {status})")

        def solve(vector):
            solution, status = solve_band(factor, bandwidth, bandwidth, vector, pivots)
            if status:
                raise ValueError(f"Banded water preconditioner solve failed (LAPACK {status})")
            return solution

    else:
        raise ValueError(f"Unknown water preconditioner: {method}")
    return LinearOperator((size, size), matvec=solve, dtype=np.complex128)


def solve_water(water, coupling, rhs, load, average, solve_ice, *, s,
                ice_density=900.0, ice_vp=3600.0, ice_vs=1800.0, ice_thickness=None, pressure_x=None,
                rtol=2e-10, preconditioner="banded", bandwidth=96, restart=40,
                maxiter=8, progress=None, gpu_device=0, gpu_library=None):
    """Apply the exact Schur action using one ice backsolve per Krylov step.

    Water unknowns are wet-face pressure followed by the outer-wall scaled
    flux. The pressure load maps into the original ice DOF ordering and
    average maps ice displacements to the ice outward normal. The water
    equation is Bx - C average(A^-1 load x_wet) = rhs.
    """
    start = perf_counter()
    size, trace_size = coupling.shape
    if (water.shape != (size, size) or load.shape[1] != trace_size
            or average.shape != (trace_size, load.shape[0])
            or np.shape(rhs) != (size,)):
        raise ValueError("Incompatible water and ice interface dimensions")
    if not (0 < rtol < 1 and bandwidth >= 0 and min(restart, maxiter) >= 1):
        raise ValueError("Invalid GMRES tolerance or iteration parameters")
    if gpu_library is None:
        workspace = nullcontext()
    else:
        from .gpu.water import WaterCUDA

        workspace = WaterCUDA(gpu_library, gpu_device, water, coupling)
    with workspace as cuda:
        inverse = (_preconditioner(water, coupling, preconditioner, bandwidth, s, ice_density=ice_density,
                                   ice_vp=ice_vp, ice_vs=ice_vs, ice_thickness=ice_thickness, pressure_x=pressure_x,
                                   cuda=cuda)
                   if isinstance(preconditioner, str) else preconditioner)
        preconditioner_s = perf_counter() - start

        def action(vector):
            normal = average @ solve_ice(load @ vector[:trace_size])
            if cuda is not None:
                return cuda.action(vector, normal)
            return water @ vector - coupling @ normal

        iterations = 0

        def record(pr_norm):
            nonlocal iterations
            iterations += 1
            if progress is not None and iterations % 20 == 0:
                progress(iterations, float(pr_norm))

        operator = LinearOperator((size, size), matvec=action, dtype=np.complex128)
        solution, status = gmres(operator, rhs, M=inverse, atol=0,
                                 rtol=rtol, restart=restart, maxiter=maxiter,
                                 callback=record, callback_type="pr_norm")
    # Independently check the host form of the original water equation.
    displacement = solve_ice(load @ solution[:trace_size])
    normal = average @ displacement
    residual = np.linalg.norm(water @ solution - coupling @ normal - rhs)
    relative = float(residual / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    if status or not np.isfinite(relative) or relative > rtol * (1 + 1e-12):
        raise WaterConvergenceError(f"Water GMRES failed at {s}: status={status}, "
                                    f"{iterations} iterations, true relative residual {relative:.3g}")
    return WaterSolution(solution, displacement, normal, relative, iterations,
                         preconditioner_s, perf_counter() - start)
