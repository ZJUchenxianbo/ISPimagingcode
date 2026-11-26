"""Compare paper TD-DSM FFT implementation against literal finite-time sums."""

import json

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
from scipy.special import hankel2

from ..ice2d.config import ImagingConfig
from ..ice2d.geometry import line_weights
from ..ice2d.imaging import dsm_energy, image_grid


def run():
    rng = np.random.default_rng(501)
    time = np.arange(24) * .0001
    source_x = np.array([-.4, .4])
    receiver_x = np.array([-.6, 0, .6])
    pulse = np.exp(-.5 * ((time - .0004) / .0001)**2) * np.cos(2 * np.pi * 2000 * (time - .0004))
    pressure, gradient = rng.normal(size=(2, 3, 24)), rng.normal(size=(2, 3, 24))
    mask = np.array([[1, 1, 0], [0, 1, 1]])
    data = dict(time=time, pressure=pressure, source_x=source_x, receiver_x=receiver_x,
                array_z=-2.5, source_equation='wave_equation_rhs', source_signal=pulse)
    processed = dict(gated_pressure=pressure, gated_gradient=gradient, receiver_mask=mask, source_signal=pulse)
    config = ImagingConfig(water_speed=1500, ice_p_speed=3600, ice_top=0,
                           source_center=.0004, band_low=1000, band_high=3000,
                           mean_bottom=-1.5, first_window_halfwidth=.0002, maximum_offset=1,
                           image_halfwidth=.1, image_x_step=.2,
                           image_z_min=-1.6, image_z_max=-1.4, image_z_step=.2)
    images = dsm_energy(processed, data, config)
    nt, dt = len(time), time[1] - time[0]
    nfft = next_fast_len(2 * nt)
    frequency = rfftfreq(nfft, dt)
    selected = np.where((frequency >= config.band_low) & (frequency <= config.band_high))[0]
    q_hat = rfft(pulse, nfft)[selected]
    k = 2 * np.pi * frequency[selected] / config.water_speed

    def kernels(receiver_z, px, pz):
        dx = receiver_x[None, :] - np.asarray(px)[:, None]
        dz = receiver_z - np.asarray(pz)[:, None]
        radius = np.hypot(dx, dz)
        g = -.25j * hankel2(0, radius[..., None] * k)
        dn_g = -.25j * k * hankel2(1, radius[..., None] * k) * (dz / radius)[..., None]
        traces = []
        for value in (g, dn_g):
            spectrum = np.zeros(value.shape[:2] + (nfft // 2 + 1,), complex)
            spectrum[..., selected] = value * q_hat
            traces.append(irfft(spectrum, nfft)[..., :nt])
        return traces

    mirror_z = 2 * config.mean_bottom - data['array_z']
    mirror_source, dn_mirror_source = kernels(mirror_z, source_x, np.full(2, data['array_z']))
    wr, ws = line_weights(receiver_x), line_weights(source_x)
    x, z = image_grid(config)
    explicit = {name: np.zeros((len(z), len(x))) for name in images}
    for iz, zz in enumerate(z):
        for ix, xx in enumerate(x):
            phi, dn_phi = [value[0] for value in kernels(data['array_z'], [xx], [zz])]
            mirror_phi, dn_mirror_phi = [value[0] for value in kernels(mirror_z, [xx], [zz])]
            v, corr = np.zeros((2, nt)), np.zeros((2, nt))
            for source in range(2):
                for receiver in range(3):
                    weight = wr[receiver] * mask[source, receiver] * dt
                    for i in range(nt):
                        for l in range(i, nt):
                            v[source, i] += weight * (
                                -gradient[source, receiver, l] * phi[receiver, l - i]
                                - pressure[source, receiver, l] * dn_phi[receiver, l - i])
                            corr[source, i] += weight * (
                                -dn_mirror_source[source, receiver, l] * mirror_phi[receiver, l - i]
                                + mirror_source[source, receiver, l] * dn_mirror_phi[receiver, l - i])
            for name, field in (('dsm', v), ('dsm_corrected', v + corr), ('correction_only', corr)):
                explicit[name][iz, ix] = dt * np.sum(ws[:, None] * field**2)
    errors = {name: float(np.linalg.norm(images[name] - explicit[name]) /
                          max(np.linalg.norm(explicit[name]), 1e-30)) for name in images}
    assert max(errors.values()) < 1e-11, errors
    zero = dict(processed, gated_pressure=pressure * 0, gated_gradient=gradient * 0)
    zero_images = dsm_energy(zero, data, config)
    assert np.all(zero_images['dsm'] < 1e-30)
    assert np.allclose(zero_images['dsm_corrected'], zero_images['correction_only'], rtol=1e-12, atol=1e-30)
    empty = dict(processed, receiver_mask=mask * 0)
    empty_images = dsm_energy(empty, data, config)
    assert all(np.all(value == 0) for value in empty_images.values())
    flipped = dict(processed, gated_gradient=-gradient)
    assert not np.allclose(dsm_energy(flipped, data, config)['dsm'], images['dsm'], atol=0)
    try:
        dsm_energy(processed, dict(data, source_equation='stress_rate_rhs'), config)
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous source units were accepted')
    return dict(passed=True, explicit_finite_sum_relative_errors=errors,
                zero_data_has_only_correction=True, zero_aperture_zero_indicator=True,
                normal_flip_detected=True, incompatible_source_rejected=True,
                physical_focusing_proved=False, neural_network='skipped')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
