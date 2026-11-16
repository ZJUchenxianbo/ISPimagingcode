"""Explicit research and inversion settings; no numerical imports for --estimate."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from .config import Config


@dataclass(frozen=True)
class InversionConfig:
    reference_top: float = 0.0
    reference_bottom: float = -1.5
    control_x: tuple = (-1.6, -0.8, 0.0, 0.8, 1.6)
    difference_step: float = 0.001
    max_iterations: int = 4
    regularization: float = 0.001
    smoothness: float = 0.01
    max_step: float = 0.03
    top_bound: float = 0.4
    bottom_bound: float = 0.2
    relative_tolerance: float = 0.001
    window: str = "full"
    internal_roundtrips: int = 2
    max_backtracks: int = 6

    def validate(self, physics):
        if self.window not in ("bottom", "internal", "full"):
            raise ValueError("window must be bottom, internal or full")
        for f in fields(self):
            if f.name not in ("window", "control_x") and not math.isfinite(getattr(self, f.name)):
                raise ValueError(f"Nonfinite inversion setting: {f.name}")
        if len(self.control_x) < 3 or not all(math.isfinite(x) for x in self.control_x):
            raise ValueError("At least three finite bottom control nodes are needed")
        if any(b <= a for a, b in zip(self.control_x, self.control_x[1:])):
            raise ValueError("control_x must be increasing")
        if max(abs(x) for x in self.control_x) >= physics.horizontal_halfwidth - physics.sponge_width:
            raise ValueError("Control nodes must lie inside the side absorbers")
        for name in ("max_iterations", "internal_roundtrips", "max_backtracks"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.difference_step, self.max_step, self.top_bound, self.bottom_bound,
               self.relative_tolerance) <= 0 or min(self.regularization, self.smoothness) < 0:
            raise ValueError("Invalid inverse bounds, steps or regularization")
        if self.reference_top <= self.reference_bottom:
            raise ValueError("Invalid reference layer")
        if physics.interface_scheme != "smooth" or physics.acquisition != "linear":
            raise ValueError("Numerical sensitivity requires smooth interfaces and linear acquisition")
        return self


@dataclass(frozen=True)
class ExperimentConfig:
    top_shift: float = 0.03
    truth_grid_ratio: float = 1.0
    noise_snr_db: float | None = None
    baselines: bool = True
    parameter_notes: str = "Empirical settings; see README for scope."

    def validate(self):
        if not math.isfinite(self.top_shift) or not math.isfinite(self.truth_grid_ratio):
            raise ValueError("Nonfinite experimental settings")
        if self.truth_grid_ratio < 1:
            raise ValueError("truth_grid_ratio must be at least one")
        if self.noise_snr_db is not None and not math.isfinite(self.noise_snr_db):
            raise ValueError("Invalid noise SNR")
        if not isinstance(self.baselines, bool) or not isinstance(self.parameter_notes, str):
            raise ValueError("Invalid baseline flag or parameter notes")
        return self


def _read(cls, values):
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    values = dict(values)
    if "control_x" in values:
        values["control_x"] = tuple(values["control_x"])
    return cls(**values)


def read_dual_config(path):
    value = json.loads(Path(path).read_text())
    if set(value) != {"physics", "inverse", "experiment"}:
        raise ValueError("Dual config requires exactly physics, inverse and experiment sections")
    physics = _read(Config, value["physics"]).validate()
    inverse = _read(InversionConfig, value["inverse"]).validate(physics)
    experiment = _read(ExperimentConfig, value["experiment"]).validate()
    return physics, inverse, experiment


def estimate_cost(physics, inverse, experiment, mode):
    h, dt = physics.grid_spacing, physics.time_step
    nx = round(2 * physics.horizontal_halfwidth / h) + 4
    nz = round((physics.computational_top + physics.seabed_depth) / h) + 4
    steps = math.ceil(physics.record_seconds / dt)
    stride = max(1, round(physics.output_interval / dt))
    receivers = math.floor(2 * physics.receiver_halfwidth / physics.receiver_spacing + 1e-9) + 1
    samples = physics.source_count * receivers * ((steps - 1) // stride + 1)
    parameters = len(inverse.control_x) + 1
    active = 1 if mode == "top" else len(inverse.control_x) if mode.startswith("bottom") else parameters
    runs = 2 + 4 * parameters + inverse.max_iterations * (2 * active + inverse.max_backtracks + 1)
    return dict(grid=[nz, nx], steps=steps, dt_seconds=dt,
                minimum_cells_per_wavelength=min(physics.air_speed if physics.upper_medium == "air" else physics.water_speed,
                                                physics.water_speed, physics.ice_s_speed) / physics.band_high / h,
                pressure_gradient_gib=8 * samples / 2**30,
                one_worker_grid_arrays_gib=4 * nx * nz * 18 / 2**30,
                derivative_disk_gib=8 * samples * parameters / 2**30,
                forward_shots_upper_estimate=runs * physics.source_count,
                truth_grid_ratio=experiment.truth_grid_ratio,
                truth_compute_scaling_approx=experiment.truth_grid_ratio ** 3,
                note="Per experiment, includes initial/final parameter differences; excludes baselines/validation, cache and plots. all runs four experiments.")
