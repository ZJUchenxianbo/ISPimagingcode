"""Explicit physical and numerical parameters. Units are SI throughout."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    water_speed: float = 1500.0
    water_density: float = 1025.0
    ice_p_speed: float = 3600.0
    ice_s_speed: float = 1800.0
    ice_density: float = 920.0
    ice_top: float = 0.0
    mean_bottom: float = -1.5
    rough_amplitude: float = 0.1
    rough_period: float = 4.0
    horizontal_halfwidth: float = 5.0
    computational_top: float = 0.8
    seabed_depth: float = 5.5
    array_depth: float = 3.0
    source_halfwidth: float = 1.8
    source_count: int = 7
    receiver_halfwidth: float = 2.4
    receiver_spacing: float = 0.04
    grid_spacing: float = 0.01
    courant: float = 0.32
    record_seconds: float = 0.0045
    output_interval: float = 0.000004
    sponge_width: float = 0.8
    sponge_rate: float = 100000.0
    source_frequency: float = 15000.0
    source_sigma: float = 0.000045
    source_center: float = 0.00027
    band_low: float = 10000.0
    band_high: float = 20000.0
    image_halfwidth: float = 1.6
    image_x_step: float = 0.04
    image_z_min: float = -1.9
    image_z_max: float = -1.1
    image_z_step: float = 0.01
    first_window_halfwidth: float = 0.00032
    imaging_aperture: float = 1.2
    maximum_offset: float = 1.6
    noise_seed: int = 20261115

    # Legacy defaults stay compatible; dual-interface configurations select air/smooth/linear.
    upper_medium: str = "free_surface"
    air_speed: float = 343.0
    air_density: float = 1.225
    interface_scheme: str = "staircase"
    interface_width_cells: float = 2.0
    acquisition: str = "snap"
    snapshots: bool = True
    spatial_order: int = 4

    @property
    def maximum_speed(self):
        return max(self.water_speed, self.ice_p_speed, self.air_speed if self.upper_medium == "air" else 0)

    @property
    def time_step(self):
        return self.courant * self.grid_spacing / self.maximum_speed

    def validate(self):
        numeric_values = [value for key, value in asdict(self).items()
                          if key not in ("upper_medium", "interface_scheme", "acquisition")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric_values):
            raise ValueError("All configuration values must be finite numbers.")
        if self.upper_medium not in ("air", "free_surface"):
            raise ValueError("upper_medium must be air or free_surface")
        if self.interface_scheme not in ("staircase", "smooth") or self.acquisition not in ("snap", "linear"):
            raise ValueError("Invalid interface_scheme or acquisition")
        if not isinstance(self.snapshots, bool):
            raise ValueError("snapshots must be boolean")
        if self.spatial_order not in (2, 4):
            raise ValueError("spatial_order must be 2 or 4")
        if self.upper_medium == "air" and self.spatial_order != 2:
            raise ValueError("Air's strong density contrast requires the nearest-neighbor second-order stencil")
        positive = (self.air_speed, self.air_density, self.interface_width_cells,
                    self.water_speed, self.water_density, self.ice_p_speed, self.ice_s_speed,
                    self.ice_density, self.rough_period, self.horizontal_halfwidth,
                    self.receiver_spacing, self.grid_spacing, self.record_seconds,
                    self.output_interval, self.sponge_width, self.sponge_rate,
                    self.source_sigma, self.source_center, self.band_low,
                    self.image_x_step, self.image_z_step, self.first_window_halfwidth,
                    self.imaging_aperture, self.maximum_offset)
        if min(positive) <= 0:
            raise ValueError("Material constants, durations, spacings, periods and apertures must be positive.")
        if not isinstance(self.source_count, int) or self.source_count < 1:
            raise ValueError("source_count must be a positive integer.")
        if not isinstance(self.noise_seed, int) or self.noise_seed < 0:
            raise ValueError("noise_seed must be a nonnegative integer.")
        if self.image_z_min >= self.image_z_max or self.image_z_min <= -self.array_depth:
            raise ValueError("The nonempty image depth interval must lie above the acquisition line.")
        if self.image_halfwidth <= 0 or self.receiver_halfwidth <= 0 or self.source_halfwidth < 0 or self.rough_amplitude < 0:
            raise ValueError("Invalid acquisition widths, imaging width, or roughness amplitude.")
        if self.record_seconds <= self.source_center + 4 * self.source_sigma:
            raise ValueError("The recording time must include the source pulse.")
        if not (self.band_low < self.source_frequency < self.band_high):
            raise ValueError("Source centre frequency must lie inside the processing band.")
        if self.water_speed / self.band_high / self.grid_spacing < 6:
            raise ValueError("Use at least six cells per shortest water wavelength.")
        speeds = [self.water_speed, self.ice_p_speed, self.ice_s_speed]
        if self.upper_medium == "air":
            speeds.append(self.air_speed)
            if self.computational_top - self.sponge_width <= self.ice_top + self.interface_width_cells * self.grid_spacing:
                raise ValueError("Air top absorber must be separated from the ice interface")
        if min(speeds) / self.band_high / self.grid_spacing < 6:
            raise ValueError("Resolve the shortest air/water/P/S wavelength with at least six cells")
        if self.time_step * max(1, round(self.output_interval / self.time_step)) >= 1 / (2 * self.band_high):
            raise ValueError("Actual rounded sampling interval violates Nyquist")
        if self.ice_p_speed ** 2 <= 2 * self.ice_s_speed ** 2:
            raise ValueError("This implementation requires positive ice lambda.")
        if not (self.seabed_depth > self.array_depth > -self.mean_bottom + self.rough_amplitude):
            raise ValueError("Require the array below the entire ice bottom and above the seabed.")
        if self.ice_top <= self.mean_bottom + self.rough_amplitude:
            raise ValueError("Ice thickness must be positive.")
        if self.computational_top <= self.ice_top:
            raise ValueError("Computational top must exceed the ice top.")
        if self.output_interval >= 1.0 / (2 * self.band_high):
            raise ValueError("Recorded sampling interval violates Nyquist.")
        if max(self.source_halfwidth, self.receiver_halfwidth) >= self.horizontal_halfwidth - self.sponge_width:
            raise ValueError("Sources and receivers must lie outside the side sponge.")
        if not 0 < self.courant <= 0.4:
            raise ValueError("Use 0 < courant <= 0.4 for the fourth-order staggered stencil.")
        return self

    def as_dict(self):
        return asdict(self)


def read_config(path=None):
    if path is None:
        return Config().validate()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Config(**values).validate()

