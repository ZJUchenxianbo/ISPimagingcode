"""Discrete coupled-interface waveform sensitivities (proposal section 4.6).

These are central differences of a regularized finite-difference forward model,
not analytic shape derivatives, a Green-tensor backpropagator, or an adjoint RTM.
"""

from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .geometry import Geometry, line_weights
from .solver import SOLVER_VERSION, simulate
from .data_store import identifier, write_manifest


class ForwardBudgetExceeded(RuntimeError):
    pass


def resample_pressure(data, target):
    """Align time only; physical source/receiver positions must agree."""
    for key in ("source_x", "receiver_x", "array_z"):
        if not np.allclose(data[key], target[key], rtol=0, atol=1e-10):
            raise ValueError(f"Incompatible physical acquisition: {key}")
    if str(data["solver_version"]) != SOLVER_VERSION:
        raise ValueError("Dual inversion requires current-version records; historical v1 is a legacy baseline only")
    source_time, time = data["time"], target["time"]
    # Small endpoint differences occur on distinct time grids. Never extrapolate:
    # the caller excludes non-overlapping samples from every objective/window.
    shape = (len(target["source_x"]), len(target["receiver_x"]), len(time))
    pressure = np.empty(shape)
    for s in range(shape[0]):
        for r in range(shape[1]):
            pressure[s, r] = np.interp(time, source_time, data["pressure"][s, r])
    valid = (time >= source_time[0]) & (time <= source_time[-1])
    return pressure, valid


class CoupledForward:
    """Known physics and reference geometry only; no truth geometry is accepted."""

    def __init__(self, physics, inverse, cache_dir, workers=1, quiet=True):
        self.physics = physics
        self.inverse = inverse
        self.cache_dir = Path(cache_dir)
        self.workers = workers
        self.quiet = quiet
        self.calls = 0
        self.budget = None
        self.parameter_count = len(inverse.control_x) + 1

    def geometry(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.parameter_count,) or not np.isfinite(theta).all():
            raise ValueError("Invalid interface parameter vector")
        return Geometry(self.inverse.reference_top + float(theta[0]), self.inverse.control_x,
                        tuple(self.inverse.reference_bottom + theta[1:])).validate(self.physics)

    def evaluate(self, theta):
        if self.budget is not None and self.calls >= self.budget:
            raise ForwardBudgetExceeded("Forward model evaluation budget exhausted")
        self.calls += 1
        return simulate(self.physics, "flat", self.cache_dir, self.workers,
                        geometry=self.geometry(theta), save=False, quiet=self.quiet)


def fixed_windows(physics, inverse, data):
    """Reference-only gates; P/S roundtrip estimates are not mode separation."""
    time = data["time"][None, None, :]
    offset = data["receiver_x"][None, :] - data["source_x"][:, None]
    height = inverse.reference_bottom - float(data["array_z"])
    bottom = physics.source_center + np.hypot(offset, 2 * height) / physics.water_speed
    thickness = inverse.reference_top - inverse.reference_bottom
    first_p = bottom + 2 * thickness / physics.ice_p_speed
    last_s = bottom + inverse.internal_roundtrips * 2 * thickness / physics.ice_s_speed
    half = physics.first_window_halfwidth

    def gate(start, stop):
        # C1 cosine shoulders of half-window/2; fixed for all iterations.
        ramp = half / 2
        left = np.clip((time - start[..., None]) / ramp, 0, 1)
        right = np.clip((stop[..., None] - time) / ramp, 0, 1)
        return (0.5 - 0.5 * np.cos(np.pi * left)) * (0.5 - 0.5 * np.cos(np.pi * right))

    return dict(bottom=gate(bottom - half, bottom + half),
                internal=gate(first_p - half, last_s + half),
                full=gate(bottom - half, np.full_like(bottom, data["time"][-1])),
                predicted_bottom=bottom, predicted_first_p=first_p, predicted_last_s=last_s)


class WaveformObjective:
    def __init__(self, physics, inverse, reference, observations):
        self.reference = reference
        self.time = reference["time"]
        dt = float(np.diff(self.time)[0])
        self.sos = butter(4, [physics.band_low, physics.band_high], fs=1 / dt,
                          btype="bandpass", output="sos")
        observed, valid = resample_pressure(observations, reference)
        self.observed = self.filter(observed)
        self.windows = fixed_windows(physics, inverse, reference)
        self.weights = (line_weights(reference["source_x"])[:, None, None]
                        * line_weights(reference["receiver_x"])[None, :, None]
                        * line_weights(self.time)[None, None, :])
        self.weights = self.weights * valid[None, None, :]
        self.window_name = inverse.window
        self.initial_residual = self.observed - self.filter(reference["pressure"])
        initial_energy = np.sum(self.initial_residual ** 2 * self.weight(inverse.window))
        reference_energy = np.sum(self.filter(reference["pressure"]) ** 2 * self.weight(inverse.window))
        self.scale = max(np.sqrt(initial_energy), np.sqrt(reference_energy) * 1e-8, 1e-20)
        if not np.any(self.weight(inverse.window) > 0):
            raise ValueError("No usable samples in the requested reference time window")

    def filter(self, pressure):
        return sosfiltfilt(self.sos, pressure, axis=-1)

    def weight(self, name=None):
        return self.weights * self.windows[name or self.window_name] ** 2

    def residual(self, predicted):
        pressure, valid = resample_pressure(predicted, self.reference)
        if not valid.all():
            raise ValueError("Predicted records must use the fixed inversion time grid")
        return self.observed - self.filter(pressure)

    def norm(self, residual, name=None):
        return float(np.sqrt(np.sum(residual ** 2 * self.weight(name))) / self.scale)


def linearize(forward, objective, theta, directory, indices=None, step=None):
    """Disk-backed columns: shape (parameter, source*receiver*time).

    Only two shot gathers and one derivative are formed at a time. Underlying
    forward shot caches include the full geometry and solver version.
    """
    if indices is None:
        indices = np.arange(forward.parameter_count)
    step = step or forward.inverse.difference_step
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = dict(physics=forward.physics.as_dict(), solver=SOLVER_VERSION,
                     geometry=forward.geometry(theta).as_dict(), controls=forward.inverse.control_x,
                     indices=list(map(int, indices)), step=step, time=objective.time.tolist())
    path = forward.cache_dir / "derivatives" / (identifier(signature) + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(indices), objective.observed.size)
    if path.exists():
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if cached.shape == shape and np.isfinite(cached).all():
            return cached
        raise ValueError("Invalid derivative cache")
    temporary = path.with_suffix(".tmp.npy")
    columns = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64,
                                       shape=(len(indices), objective.observed.size))
    for row, parameter in enumerate(indices):
        plus, minus = np.array(theta, dtype=float), np.array(theta, dtype=float)
        plus[parameter] += step
        minus[parameter] -= step
        upper = forward.evaluate(plus)
        upper_filtered = objective.filter(upper["pressure"])
        del upper
        lower = forward.evaluate(minus)
        columns[row] = ((upper_filtered - objective.filter(lower["pressure"])) / (2 * step)).ravel()
        if not np.isfinite(columns[row]).all():
            raise FloatingPointError("Nonfinite interface sensitivity")
    columns.flush()
    del columns
    temporary.replace(path)
    write_manifest(path.with_suffix(".json"), signature)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def normal_system(columns, residual, objective, name=None):
    weight = objective.weight(name).ravel() / objective.scale ** 2
    rhs = np.empty(len(columns))
    gram = np.empty((len(columns), len(columns)))
    for i in range(len(columns)):
        weighted = columns[i] * weight
        rhs[i] = np.dot(weighted, residual.ravel())
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = np.dot(weighted, columns[j])
    return gram, rhs


def sensitivity_diagnostics(columns, residual, objective, name=None):
    gram, rhs = normal_system(columns, residual, objective, name)
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    safe = np.maximum(norms, np.finfo(float).eps * max(float(norms.max()), 1))
    correlation = gram / safe[:, None] / safe[None, :]
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(gram)[::-1], 0))
    normalized = np.sqrt(np.maximum(np.linalg.eigvalsh(correlation)[::-1], 0))
    # Gram construction loses precision below sqrt(machine epsilon).
    rank = int(np.sum(singular > max(float(singular[0]) * 1e-7, 1e-14)))
    return dict(column_norms=norms, singular_values=singular,
                normalized_singular_values=normalized, correlation=correlation,
                signed_indicator=rhs / safe, indicator=np.abs(rhs / safe), rank=rank,
                max_top_bottom_correlation=float(np.max(np.abs(correlation[0, 1:]))),
                residual_norm=objective.norm(residual, name))
