"""First-reflection Cauchy traces, with one source convention and one bandpass."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .io import validate_cauchy


def process_records(data, config):
    validate_cauchy(data)
    dt = float(np.diff(data['time'])[0])
    if config.band_high >= .5 / dt:
        raise ValueError('Imaging band violates the recorded Nyquist frequency')
    sos = butter(4, [config.band_low, config.band_high], fs=1 / dt,
                 btype='bandpass', output='sos')
    pressure = sosfiltfilt(sos, data['pressure_scattered'], axis=-1)
    gradient = sosfiltfilt(sos, data['gradient_scattered'], axis=-1)
    source = sosfiltfilt(sos, data['source_signal'])
    offset = data['receiver_x'][None, :] - data['source_x'][:, None]
    height = config.mean_bottom - float(data['array_z'])
    reference_arrival = config.source_center + np.hypot(offset, 2 * height) / config.water_speed
    difference = data['time'][None, None, :] - reference_arrival[..., None]
    half = config.first_window_halfwidth
    ramp = np.clip((abs(difference) - .75 * half) / (.25 * half), 0, 1)
    window = .5 * (1 + np.cos(np.pi * ramp))
    mask = (abs(offset) <= config.maximum_offset).astype(float)
    if not np.all(np.sum(mask, axis=1) >= 2):
        raise ValueError('Each source needs at least two receivers within maximum_offset')
    return dict(pressure=pressure, gradient=gradient, gated_pressure=pressure * window,
                gated_gradient=gradient * window, source_signal=source,
                window=window, reference_arrival=reference_arrival, receiver_mask=mask)
