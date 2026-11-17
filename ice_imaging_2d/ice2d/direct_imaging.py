"""Proposal 4.3 only: weighted Cauchy adjoint backpropagation.

V_s(z,t) = sum_r w_r int [dn d_s(r,tau) conj G(r,z,tau-t)
                         - d_s(r,tau) conj dn_r G(r,z,tau-t)] d tau.
The advanced kernel is explicitly the adjoint of the supplied causal kernel.
No interface perturbation, fitting, correction field or upper indicator occurs.
"""
import numpy as np
from scipy.signal import correlate, correlation_lags
from .background_green import uniform_step
from .geometry import line_weights


def cauchy_backpropagate(pressure, normal_pressure, time, green):
    green.validate()
    pressure = np.asarray(pressure)
    normal_pressure = np.asarray(normal_pressure)
    if pressure.ndim != 3 or pressure.shape != normal_pressure.shape:
        raise ValueError('Cauchy traces must have shape (source, receiver, time)')
    if pressure.shape[1:] != (len(green.receiver_x), len(time)):
        raise ValueError('Cauchy trace coordinates do not match kernels')
    if not np.isfinite(pressure).all() or not np.isfinite(normal_pressure).all():
        raise ValueError('Nonfinite Cauchy traces')
    dt = uniform_step(time)
    if not np.isclose(dt, uniform_step(green.lag), rtol=1e-8, atol=1e-14):
        raise ValueError('Data and kernels must use identical time steps; no implicit resampling')
    weights = line_weights(time)
    receiver_weights = line_weights(green.receiver_x)
    lags = correlation_lags(len(time), len(green.lag), mode='full')
    result = np.zeros((pressure.shape[0], len(green.points), len(lags)),
                      dtype=np.result_type(pressure, green.gpp, green.normal_gpp, np.float64))
    # Independent sampling points share the same fixed background. No forward
    # solver is called here. Full linear correlation avoids circular FFT wrap.
    for sample in range(len(green.points)):
        for source in range(pressure.shape[0]):
            for receiver, weight in enumerate(receiver_weights):
                first = correlate(normal_pressure[source, receiver] * weights,
                                  green.gpp[sample, receiver], mode='full', method='fft')
                second = correlate(pressure[source, receiver] * weights,
                                   green.normal_gpp[sample, receiver], mode='full', method='fft')
                result[source, sample] += weight * (first - second)
    return dict(field=result, time=time[0] + lags * dt, points=green.points)
