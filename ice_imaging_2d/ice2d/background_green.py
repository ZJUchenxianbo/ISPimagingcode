"""Contract for externally validated coupled-background impulse kernels.

This module validates kernels; it does not claim to construct the missing
coupled Green tensor or its elastic/air traces from the proposal.
"""
import json
from dataclasses import dataclass
import numpy as np
from .data_store import identifier


@dataclass(frozen=True)
class PressureGreen:
    gpp: np.ndarray                  # (sample, receiver, lag)
    normal_gpp: np.ndarray           # derivative at receiver, same shape
    lag: np.ndarray                 # causal impulse time, starts at zero
    points: np.ndarray              # (sample, [x,z]), water points only
    receiver_x: np.ndarray
    array_z: float
    metadata: dict

    def validate(self):
        shape = self.gpp.shape
        if len(shape) != 3 or shape != self.normal_gpp.shape or min(shape) < 1:
            raise ValueError('Gpp and receiver-normal Gpp must have shape (sample, receiver, lag)')
        if self.points.shape != (shape[0], 2) or self.receiver_x.shape != (shape[1],):
            raise ValueError('Kernel coordinate dimensions do not match')
        uniform_step(self.lag)
        if self.lag[0] != 0 or len(self.lag) != shape[2]:
            raise ValueError('Impulse kernel lag must start at zero and match the kernel')
        for value in (self.gpp, self.normal_gpp, self.points, self.receiver_x, np.array(self.array_z)):
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite kernel or coordinate')
        required = dict(schema='ice-cauchy-green-v1', response='causal_impulse',
                        normal='positive_z', background='air-elastic_ice-water',
                        derivative='receiver_normal', sample_medium='water',
                        source_equation='pressure_rate_rhs')
        for key, value in required.items():
            if self.metadata.get(key) != value:
                raise ValueError(f'Kernel metadata requires {key}={value}')
        if not self.metadata.get('reference_signature') or not self.metadata.get('validation_report'):
            raise ValueError('Kernel needs reference signature and validation report provenance')
        return self


def uniform_step(time):
    time = np.asarray(time)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError('At least two finite time samples required')
    dt = time[1] - time[0]
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-8, atol=1e-14):
        raise ValueError('Time samples must be uniformly increasing')
    return float(dt)


def reference_signature(reference):
    return identifier(dict(physics=json.loads(str(reference['config_json'])),
                           geometry=json.loads(str(reference['geometry_json']))))


def load_pressure_green(path):
    with np.load(path, allow_pickle=False) as value:
        kernel = PressureGreen(value['gpp'], value['normal_gpp'], value['lag'],
                               value['points'], value['receiver_x'], float(value['array_z']),
                               json.loads(str(value['metadata_json'])))
    return kernel.validate()
