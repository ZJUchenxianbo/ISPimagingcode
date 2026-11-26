"""CQ physics, geometry, acquisition and numerical parameters."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from ..data_store import identifier
from . import SOLVER_VERSION


DEFAULTS = {
    'geometry': dict(width_m=10.0, water_depth_m=5.5, pml_x_m=0.8, pml_z_m=0.8,
                     top='0.0', bottom='-1.5'),
    'materials': dict(water_speed=1500.0, water_density=1025.0, ice_density=920.0,
                      ice_p_speed=3600.0, ice_s_speed=1800.0, air_speed=343.0, air_density=1.225),
    'acquisition': dict(sources=[[-0.3, -2.0]], receivers=[[0.4, -2.0]]),
    'frequency': dict(carrier_hz=15000.0, min_hz=500.0, max_hz=35000.0,
                      duration_s=0.0045, dt_s=0.000004, sigma_per_s=4.8 / 0.0045, samples=None),
    'pulse': dict(kind='gaussian', sigma_s=0.000045, center_s=0.00027, amplitude=1.0),
    'processing': dict(band_low=10000.0, band_high=20000.0),
    'solver': dict(backend='gpu', air='zero-traction', native=False, mesh_scale=1.0,
                   water_panel_m=None, ice_elements_per_wavelength=3.0,
                   gpu_devices=[0], workers_per_device=1, gpu_library_dir=None,
                   cuda_arch=None),
}


@dataclass(frozen=True)
class ForwardConfig:
    values: dict

    @classmethod
    def read(cls, path):
        supplied = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls.from_dict(supplied)

    @classmethod
    def from_dict(cls, supplied):
        values = deepcopy(DEFAULTS)
        unknown = set(supplied) - set(values)
        if unknown:
            raise ValueError(f'Unknown CQ sections: {sorted(unknown)}; use a CQ JSON config')
        for section, entries in supplied.items():
            if not isinstance(entries, dict) or set(entries) - set(values[section]):
                raise ValueError(f'Unknown CQ options in {section}')
            values[section].update(entries)
        return cls(values).validate()

    def validate(self):
        v = self.values
        geometry, materials, solver = v['geometry'], v['materials'], v['solver']
        for name in ('width_m', 'water_depth_m', 'pml_x_m', 'pml_z_m'):
            if not math.isfinite(geometry[name]) or geometry[name] <= 0:
                raise ValueError(f'geometry.{name} must be positive')
        if any(not math.isfinite(x) or x <= 0 for x in materials.values()):
            raise ValueError('Material constants must be finite and positive')
        if materials['ice_p_speed']**2 <= 2 * materials['ice_s_speed']**2:
            raise ValueError('Positive ice lambda is required')
        if not all(isinstance(geometry[name], str) for name in ('top', 'bottom')):
            raise ValueError('CQ top and bottom must be numeric expressions in x')
        if solver['backend'] not in ('cpu', 'gpu') or solver['air'] not in ('zero-traction', 'coupled'):
            raise ValueError('Invalid backend or air boundary')
        if solver['backend'] == 'gpu' and solver['air'] != 'zero-traction':
            raise ValueError('CUDA supports zero-traction only; use CPU for explicit air coupling')
        if type(solver['native']) is not bool:
            raise ValueError('solver.native must be boolean')
        for name in ('mesh_scale', 'ice_elements_per_wavelength', 'water_panel_m'):
            number = solver[name]
            if number is not None and (not math.isfinite(number) or number <= 0):
                raise ValueError(f'solver.{name} must be positive')
        devices = solver['gpu_devices']
        if (not isinstance(devices, list) or not devices
                or any(type(i) is not int or i < 0 for i in devices) or len(set(devices)) != len(devices)):
            raise ValueError('gpu_devices must contain distinct nonnegative device indices')
        if type(solver['workers_per_device']) is not int or solver['workers_per_device'] < 1:
            raise ValueError('workers_per_device must be positive')
        sources, receivers = self.sources, self.receivers
        for name, points in (('sources', sources), ('receivers', receivers)):
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                raise ValueError(f'{name} must be a finite nonempty (n, 2) array')
        if not np.allclose(receivers[:, 1], receivers[0, 1], rtol=0, atol=1e-12):
            raise ValueError('The current imaging adapter requires a horizontal receiver array')
        if not np.allclose(sources[:, 1], receivers[0, 1], rtol=0, atol=1e-12):
            raise ValueError('Current imaging kernels require sources and receivers at the same depth')
        for points in (sources, receivers):
            if len(points) > 1 and np.any(np.diff(points[:, 0]) <= 0):
                raise ValueError('Acquisition x coordinates must be strictly increasing for line quadrature')
        if np.any(np.linalg.norm(sources[:, None] - receivers[None], axis=-1) < 1e-12):
            raise ValueError('Source/receiver coincidence has singular direct pressure; use distinct points')
        pulse, frequency = v['pulse'], v['frequency']
        processing = v['processing']
        low = frequency['min_hz'] if processing['band_low'] is None else processing['band_low']
        high = frequency['max_hz'] if processing['band_high'] is None else processing['band_high']
        if (not np.isfinite([low, high]).all() or not 0 < low < high
                or low < frequency['min_hz'] or high > frequency['max_hz']):
            raise ValueError('Processing band must be positive and contained in the CQ reconstruction band')
        if (pulse['kind'] not in ('gaussian', 'raised_cosine')
                or not math.isfinite(pulse['amplitude']) or pulse['amplitude'] == 0):
            raise ValueError('Invalid source pulse')
        if pulse['kind'] == 'gaussian' and not (
                math.isfinite(pulse['sigma_s']) and math.isfinite(pulse['center_s'])
                and 0 < pulse['sigma_s'] <= pulse['center_s'] / 4
                and pulse['center_s'] + 4 * pulse['sigma_s'] < frequency['duration_s']):
            raise ValueError('Gaussian pulse needs 4-sigma margins before centre and record end')
        return self

    def signature(self):
        values = deepcopy(self.values)
        values.pop('processing')
        for key in ('backend', 'native', 'gpu_devices', 'workers_per_device', 'gpu_library_dir', 'cuda_arch'):
            values['solver'].pop(key)
        return identifier(dict(solver_version=SOLVER_VERSION, forward=values))

    @property
    def sources(self):
        return self._points(self.values['acquisition']['sources'])

    @property
    def receivers(self):
        return self._points(self.values['acquisition']['receivers'])

    @staticmethod
    def _points(specification):
        if isinstance(specification, dict):
            if (set(specification) != {'x_min', 'x_max', 'count', 'z'}
                    or type(specification['count']) is not int or specification['count'] < 1):
                raise ValueError('Array ranges need x_min, x_max, positive integer count, z')
            x = np.linspace(specification['x_min'], specification['x_max'], specification['count'])
            return np.column_stack((x, np.full(len(x), specification['z'])))
        return np.asarray(specification, dtype=float)

    def case(self, *, cauchy=True):
        from .coupled import Case
        g, solver = self.values['geometry'], self.values['solver']
        return Case(g['width_m'], g['water_depth_m'], g['pml_x_m'], g['pml_z_m'],
                    g['top'], g['bottom'], tuple(self.sources[0]), tuple(self.receivers[0]),
                    self.values['frequency']['carrier_hz'], solver['air'], solver['mesh_scale'],
                    solver['ice_elements_per_wavelength'], solver['water_panel_m'],
                    **self.values['materials'], native=solver['native'], cauchy=cauchy)

    def grid(self):
        from .frequency import CQGrid
        frequency = self.values['frequency'].copy()
        if frequency['samples'] is None:
            del frequency['samples']
        return CQGrid.from_config(frequency, pulse=self.values['pulse'])

    def summary(self):
        from .coupled import PlanarTransfer
        grid = self.grid()
        case = self.case()
        # mesh_only builds coordinates/counts, never matrices or CUDA handles.
        model = PlanarTransfer(case, grid.design_hz, mesh_only=True)
        geometry = model.geometry
        mesh = model.mesh_summary()
        ns, nr, nt, nf = len(self.sources), len(self.receivers), grid.n_out, len(grid.selected)
        return dict(backend=self.values['solver']['backend'], cq_points=len(grid.selected),
                    native_band_points=len(grid.indices), sources=len(self.sources), receivers=len(self.receivers),
                    time_samples=grid.n_out, design_hz=grid.design_hz,
                    interface_panels=len(geometry.edges) - 1, ice_layers=geometry.z_layers,
                    mesh=mesh,
                    one_dense_water_complex_matrix_gib=16 * mesh['water_panels']**2 / 2**30,
                    source_projection_relative_error=self.source_projection_error(grid),
                    stored_cauchy_arrays_gib=6 * ns * nr * nt * 8 / 2**30,
                    two_complex_transfer_arrays_gib=4 * ns * nr * nf * 16 / 2**30,
                    memory_note='Array sizes only; PDE factors, FFT workspaces and worker copies are additional',
                    lower_boundary='open_water_pml', gradient_normal='positive_z')

    def source_projection_error(self, grid=None):
        """Detect contour-band truncation amplified by inverse-Laplace damping."""
        pulse = self.values['pulse']
        if pulse['kind'] != 'gaussian':
            return None
        grid = self.grid() if grid is None else grid
        time = grid.time_s[:grid.n_out]
        shift = time - pulse['center_s']
        intended = (pulse['amplitude'] * np.exp(-0.5 * (shift / pulse['sigma_s'])**2)
                    * np.cos(2 * np.pi * grid.carrier_hz * shift))
        projected = grid.waveform(np.ones(len(grid.selected), complex))
        return float(np.linalg.norm(projected - intended) / max(np.linalg.norm(intended), 1e-30))
