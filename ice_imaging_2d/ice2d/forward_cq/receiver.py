"""Receiver +z derivatives of 2D layer potentials, sharing the boundary solve.

The normal is the receiver direction +e_z, not the source-panel normal in K.
Stacking [p, dp/dz] evaluation rows lets CPU and resident CUDA solvers use the
same boundary solution. All derivative evaluation is currently on the CPU.
"""

import numpy as np
from scipy.special import kv

from .water import PlanarFluid, _quadrature


def _derivative_integrands(wave, dx, dz, ds, nx, nz):
    radius = np.sqrt(dx**2 + dz**2)
    if np.any(abs(radius) < np.finfo(float).eps):
        raise ValueError("Receiver derivative is undefined on the BEM contour")
    k0, k1 = kv(0, wave * radius), kv(1, wave * radius)
    # d = source - receiver. d_z K0(k r) = k K1(k r) d_z/r.
    single = wave * k1 * dz / radius * ds
    dot = dx * nx + dz * nz
    double = (wave * k1 * nz / radius
              - (wave**2 * k0 + 2 * wave * k1 / radius) * dot * dz / radius**2)
    return single, double


def normal_rows(fluid, s, targets, double_columns):
    """Analytically differentiate receiver rows with near-panel quadrature."""
    targets = np.atleast_2d(targets)
    tx, tz = fluid._transform(targets, s)
    wave = s / fluid.sound_speed
    # Match the pressure evaluator's far-panel quadrature exactly. Near panels
    # use the same sinh quadrature as pressure; differentiating the continuous
    # integrand then agrees with an independent receiver-coordinate difference.
    nodes, weights = _quadrature(4)
    x, z, ds, nx, nz = fluid._geometry(slice(None), nodes, s)
    single = np.empty((len(targets), fluid.panel_count), dtype=complex)
    double = np.empty((len(targets), double_columns), dtype=complex)
    for first in range(0, len(targets), 8):
        block = slice(first, first + 8)
        values, derivatives = _derivative_integrands(
            wave, x[None] - tx[block, None, None], z[None] - tz[block, None, None],
            ds[None], nx[None], nz[None])
        single[block] = np.sum(values * weights, axis=-1) / (2 * np.pi)
        double[block] = np.sum(derivatives[:, :double_columns] * weights, axis=-1) / (2 * np.pi)
    rows, columns = fluid._correction_pairs(targets)
    if len(rows):
        nodes, weights = fluid._receiver_pairs(targets, rows, columns)
        x, z, ds, nx, nz = fluid._geometry(columns, nodes, s)
        values, derivatives = _derivative_integrands(
            wave, x - tx[rows, None], z - tz[rows, None], ds, nx, nz)
        single[rows, columns] = np.sum(values * weights, axis=-1) / (2 * np.pi)
        keep = columns < double_columns
        double[rows[keep], columns[keep]] = np.sum(derivatives[keep] * weights[keep], axis=-1) / (2 * np.pi)
    if not np.isfinite(single).all() or not np.isfinite(double).all():
        raise FloatingPointError("Nonfinite receiver normal derivative")
    return single, double


class CauchyFluid(PlanarFluid):
    cauchy = False

    def assemble(self, s, receiver=None, *, double_columns=None, timings=None):
        result = super().assemble(s, receiver, double_columns=double_columns, timings=timings)
        if self.cauchy and receiver is not None:
            count = self.panel_count if double_columns is None else double_columns
            single, double = normal_rows(self, s, receiver, count)
            # Always return a row axis in Cauchy mode, even for a single point.
            result['S_receiver'] = np.vstack((result['S_receiver'], single))
            result['K_receiver'] = np.vstack((result['K_receiver'], double))
        return result

    def _device_corrections(self, s, target):
        corrections, single, double, elapsed, receiver_elapsed = super()._device_corrections(s, target)
        if self.cauchy and target is not None:
            from time import perf_counter
            start = perf_counter()
            ds, dk = normal_rows(self, s, target, self.surface_count)
            single, double = np.vstack((single, ds)), np.vstack((double, dk))
            receiver_elapsed += perf_counter() - start
        return corrections, single, double, elapsed, receiver_elapsed

    def assemble_device(self, s, receiver, *, density, timings=None):
        # Avoid the parent scalar-row squeeze after stacking Cauchy rows.
        points = np.atleast_2d(receiver) if self.cauchy and receiver is not None else receiver
        return super().assemble_device(s, points, density=density, timings=timings)

    def incident(self, s, source, receiver=None):
        boundary, observed = super().incident(s, source, receiver)
        if not self.cauchy or receiver is None:
            return boundary, observed
        targets = np.atleast_2d(receiver)
        # Only physical water receivers are allowed by PlanarTransfer.
        dx = targets[:, 0] - source[0]
        dz = targets[:, 1] - source[1]
        radius = np.hypot(dx, dz)
        wave = s / self.sound_speed
        gradient = -wave * kv(1, wave * radius) * dz / radius / (2 * np.pi)
        return boundary, np.r_[np.atleast_1d(observed), gradient]
