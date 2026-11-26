"""One formal experiment definition; inversion has no PDE implementation dependency."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np

from .forward_cq.config import ForwardConfig
from .geometry import expression


@dataclass(frozen=True)
class ImagingConfig:
    water_speed: float
    ice_p_speed: float
    ice_top: float
    source_center: float
    band_low: float
    band_high: float
    mean_bottom: float
    image_halfwidth: float
    image_x_step: float
    image_z_min: float
    image_z_max: float
    image_z_step: float
    first_window_halfwidth: float
    maximum_offset: float

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('Imaging parameters must be finite')
        positive = ('water_speed', 'ice_p_speed', 'source_center', 'band_low', 'band_high',
                    'image_halfwidth', 'image_x_step', 'image_z_step',
                    'first_window_halfwidth', 'maximum_offset')
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError('Imaging steps, band, window and aperture must be positive')
        if self.image_z_min >= self.image_z_max or self.band_low >= self.band_high:
            raise ValueError('Invalid sampling interval or processing band')
        return self


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, values):
        return cls(deepcopy(values)).validate()

    @property
    def forward(self):
        return ForwardConfig.from_dict(self.values['forward'])

    def for_case(self, name):
        values = deepcopy(self.forward.values)
        case, roughness = self.values['cases'][name], self.values['roughness']
        bottom = float(values['geometry']['top']) - case['mean_thickness_m']
        profile = f'{bottom:.17g}'
        if case['kind'] == 'sinusoidal':
            profile += f" + {roughness['height_m'] / 2:.17g}*sin(2*pi*x/{roughness['horizontal_period_m']:.17g})"
        values['geometry']['bottom'] = profile
        return ForwardConfig.from_dict(values)

    def imaging(self):
        forward = self.forward.values
        values = {key: value for key, value in self.values['imaging'].items() if key != 'window_factors'}
        values.update(water_speed=forward['materials']['water_speed'],
                      ice_p_speed=forward['materials']['ice_p_speed'],
                      ice_top=float(forward['geometry']['top']),
                      source_center=forward['pulse']['center_s'],
                      band_low=forward['processing']['band_low'],
                      band_high=forward['processing']['band_high'])
        return ImagingConfig(**values).validate()

    def validate(self):
        if set(self.values) != {'forward', 'roughness', 'cases', 'imaging'}:
            raise ValueError('Experiment needs forward, roughness, cases and imaging sections')
        if 'bottom' in self.values['forward'].get('geometry', {}):
            raise ValueError('Define bottom geometry through cases, not a second geometry.bottom setting')
        forward = self.forward
        frequency, processing = forward.values['frequency'], forward.values['processing']
        if (frequency['carrier_hz'] != 15000 or processing['band_low'] != 10000
                or processing['band_high'] != 20000):
            raise ValueError('Formal experiments require 15 kHz carrier and 10-20 kHz imaging band')
        if forward.values['pulse']['kind'] != 'gaussian':
            raise ValueError('First-reflection experiments require a short Gaussian-modulated pulse')
        if frequency['samples'] is not None:
            raise ValueError('Formal waveform experiments require all in-band CQ points')
        grid = forward.grid()  # FFT grid only, no PDE or CUDA import
        projection_error = forward.source_projection_error(grid)
        if projection_error > .01:
            raise ValueError(f'CQ source distortion {projection_error:.2%} exceeds 1%; widen reconstruction band')
        try:
            top = float(forward.values['geometry']['top'])
        except (ValueError, TypeError) as error:
            raise ValueError('This stage uses a prescribed flat ice top') from error
        roughness = self.values['roughness']
        if (set(roughness) != {'definition', 'height_m', 'horizontal_period_m'}
                or roughness['definition'] != 'vertical_peak_to_valley'
                or roughness['height_m'] != .2 or not np.isfinite(roughness['horizontal_period_m'])
                or roughness['horizontal_period_m'] <= 0):
            raise ValueError('Declare 0.20 m vertical peak-to-valley roughness and a separate horizontal period')
        if not self.values['cases']:
            raise ValueError('Select flat or sinusoidal cases; ice ridges are excluded')
        names = {field.name for field in fields(ImagingConfig)}
        derived = {'water_speed', 'ice_p_speed', 'ice_top', 'source_center', 'band_low', 'band_high'}
        if set(self.values['imaging']) != (names - derived) | {'window_factors'}:
            raise ValueError('Unknown or missing imaging settings')
        image = self.imaging()
        factors = self.values['imaging']['window_factors']
        if not factors or 1.0 not in factors or not np.isfinite(factors).all() or min(factors) <= 0:
            raise ValueError('window_factors must be positive, finite and contain 1.0')
        if image.image_z_min <= forward.receivers[0, 1]:
            raise ValueError('Sampling region must lie above the measurement line')
        if image.image_z_max >= top:
            raise ValueError('Bottom sampling region must lie below the prescribed ice top')
        offsets = abs(forward.sources[:, None, 0] - forward.receivers[None, :, 0])
        if np.any(np.sum(offsets <= image.maximum_offset, axis=1) < 2):
            raise ValueError('Each source needs at least two receivers within maximum_offset')
        g = forward.values['geometry']
        if image.image_halfwidth >= g['width_m'] / 2:
            raise ValueError('Bottom sampling region must stay within the physical lateral domain')
        points = np.linspace(-g['width_m'] / 2, g['width_m'] / 2, 4097)
        for name, case in self.values['cases'].items():
            if (not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)
                    or set(case) != {'kind', 'mean_thickness_m'}
                    or case['kind'] not in ('flat', 'sinusoidal')):
                raise ValueError('Case names must be identifiers; allowed kinds: flat, sinusoidal')
            mean = case['mean_thickness_m']
            amplitude = .1 if case['kind'] == 'sinusoidal' else 0
            if not np.isfinite(mean) or mean - amplitude <= 0 or mean + amplitude > 4:
                raise ValueError(f'{name}: every ice thickness must satisfy 0 < H <= 4 m')
            config = self.for_case(name)
            bottom = expression(config.values['geometry']['bottom'], points)
            if np.any(bottom <= -g['water_depth_m']):
                raise ValueError(f'{name}: ice bottom reaches water truncation depth')
            for label, acquisition in (('source', config.sources), ('receiver', config.receivers)):
                bottom_at_points = expression(config.values['geometry']['bottom'], acquisition[:, 0])
                if (np.any(abs(acquisition[:, 0]) >= g['width_m'] / 2)
                        or np.any(acquisition[:, 1] <= -g['water_depth_m'])
                        or np.any(acquisition[:, 1] >= bottom_at_points)):
                    raise ValueError(f'{name}: {label} points must stay in physical water')
        height = image.mean_bottom - forward.receivers[0, 1]
        if height <= 0 or image.mean_bottom >= top:
            raise ValueError('Reference bottom must lie between the measurement line and ice top')
        latest = (image.source_center + np.hypot(image.maximum_offset, 2 * height) / image.water_speed
                  + image.first_window_halfwidth * max(factors))
        if latest >= frequency['duration_s']:
            raise ValueError('Record ends before the prescribed first-reflection window')
        if image.first_window_halfwidth * max(factors) >= 2 * (top - image.mean_bottom) / image.ice_p_speed:
            raise ValueError('First-reflection window includes the nominal first ice P return')
        return self
