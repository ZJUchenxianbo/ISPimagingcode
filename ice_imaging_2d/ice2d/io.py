"""不允许 pickle 的实验数组读取。"""

import numpy as np


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_cauchy(data):
    if str(data.get('source_equation', '')) != 'wave_equation_rhs':
        raise ValueError('Data must declare wave_equation_rhs source units')
    if str(data.get('gradient_normal', '')) != 'positive_z':
        raise ValueError('Recorded gradient must explicitly declare the +z receiver derivative')
    for name in ('source_x', 'receiver_x', 'time', 'source_signal'):
        value = np.asarray(data[name])
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be a finite vector')
    for name in ('source_x', 'receiver_x', 'time'):
        if np.any(np.diff(data[name]) <= 0):
            raise ValueError(f'{name} must be strictly increasing')
    time = data['time']
    if len(time) < 32 or time[0] != 0 or not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-9, atol=1e-14):
        raise ValueError('Cauchy data needs at least 32 uniform samples starting at zero')
    if data['source_signal'].shape != time.shape:
        raise ValueError('Source waveform must match the time axis')
    shape = (len(data['source_x']), len(data['receiver_x']), len(time))
    for name in ('pressure', 'gradient', 'pressure_direct', 'gradient_direct',
                 'pressure_scattered', 'gradient_scattered'):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
    for name in ('pressure', 'gradient'):
        if not np.allclose(data[name], data[name + '_direct'] + data[name + '_scattered'], rtol=1e-10, atol=1e-14):
            raise ValueError(f'{name} does not equal direct plus scattered components')
    if not np.isfinite(float(data['array_z'])):
        raise ValueError('Invalid receiver array depth')
    return data


