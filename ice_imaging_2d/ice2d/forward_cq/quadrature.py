"""CQ trapezoidal-rule samples and source envelope; no axisymmetric solver."""
import math
import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len

def quadrature_grid(dt, duration, sigma, frequency, span, *, pulse=None):
    n_out = round(duration / dt)
    if n_out < 3 or not math.isclose(n_out * dt, duration, rel_tol=1e-12):
        raise ValueError("duration must be a whole number of sample steps")
    count = next_fast_len(2 * n_out)
    t = dt * np.arange(count)
    bins = fftfreq(count, dt)
    zeta = np.exp(-sigma * dt - 2j * np.pi * bins * dt)
    s = 2.0 * (1.0 - zeta) / ((1.0 + zeta) * dt) + 2j * np.pi * frequency
    taper = np.clip((t - duration) / (0.12 * duration), 0.0, 1.0)
    taper = 0.5 + 0.5 * np.cos(np.pi * taper)
    if pulse is None or pulse['kind'] == 'raised_cosine':
        ramp = 5.0 / frequency
        envelope = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / ramp, 0, 1)))
    else:
        width, centre = pulse['sigma_s'], pulse['center_s']
        envelope = np.exp(-0.5 * ((t - centre) / width)**2)
        envelope = envelope * np.exp(-2j * np.pi * frequency * centre)
    envelope = envelope * (pulse.get('amplitude', 1.0) if pulse else 1.0)
    source_spectrum = fft(envelope * taper * np.exp(-sigma * t))
    indices = np.flatnonzero((abs(bins) <= span) & (s.imag > 0))
    return n_out, t, bins, s, source_spectrum, indices

def band_window(bins, span):
    return 0.5 + 0.5 * np.cos(np.pi * np.clip((abs(bins) / span - 0.8) / 0.2, 0.0, 1.0))

def benchmark_indices(indices, bins, count):
    """Sample signed frequency shifts while keeping both band edges if count permits."""
    ordered = indices[np.argsort(bins[indices])]
    if count == 1:
        return ordered[[np.argmin(abs(bins[ordered]))]]
    offsets = np.rint(np.linspace(0, len(ordered) - 1, min(count, len(ordered)))).astype(int)
    return ordered[offsets]
