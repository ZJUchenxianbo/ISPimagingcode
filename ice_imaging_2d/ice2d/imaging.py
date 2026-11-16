"""Independent inversion kernels. None of these functions receives the true surface."""

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft, rfftfreq
from scipy.special import hankel2
from numba import njit
from .geometry import line_weights


def image_grid(config):
    x = np.arange(-config.image_halfwidth, config.image_halfwidth + config.image_x_step / 2, config.image_x_step)
    z = np.arange(config.image_z_min, config.image_z_max + config.image_z_step / 2, config.image_z_step)
    return x, z


@njit(cache=True)
def _delay_sum(analytic, source_x, receiver_x, array_z, time0, dt,
               image_x, image_z, speed, pulse_centre, aperture, max_offset):
    incoherent = np.zeros((len(image_z), len(image_x)))
    coherent = np.zeros_like(incoherent)
    coverage = np.zeros_like(incoherent)
    for iz in range(len(image_z)):
        for ix in range(len(image_x)):
            qx, qz = image_x[ix], image_z[iz]
            envelope_sum, complex_sum, weight_sum = 0.0, 0.0j, 0.0
            for s in range(len(source_x)):
                if abs(source_x[s] - qx) >= aperture:
                    continue
                source_weight = np.cos(np.pi * (source_x[s] - qx) / (2 * aperture)) ** 2
                source_distance = np.hypot(qx - source_x[s], qz - array_z)
                for r in range(len(receiver_x)):
                    if abs(receiver_x[r] - qx) >= aperture or abs(receiver_x[r] - source_x[s]) > max_offset:
                        continue
                    weight = source_weight * np.cos(np.pi * (receiver_x[r] - qx) / (2 * aperture)) ** 2
                    distance = source_distance + np.hypot(qx - receiver_x[r], qz - array_z)
                    index = ((distance / speed + pulse_centre) - time0) / dt
                    left = int(np.floor(index))
                    if left < 0 or left + 1 >= analytic.shape[-1]:
                        continue
                    fraction = index - left
                    value = (1 - fraction) * analytic[s, r, left] + fraction * analytic[s, r, left + 1]
                    envelope_sum += weight * abs(value)
                    complex_sum += weight * value
                    weight_sum += weight
            if weight_sum > 0:
                incoherent[iz, ix] = envelope_sum / weight_sum
                coherent[iz, ix] = abs(complex_sum) / weight_sum
                coverage[iz, ix] = weight_sum
    return incoherent, coherent, coverage


def migrate(processed, data, config, speed_factor=1.0, aperture_factor=1.0, receiver_stride=1):
    x, z = image_grid(config)
    values = _delay_sum(
        np.ascontiguousarray(processed["analytic"][:, ::receiver_stride]), data["source_x"],
        data["receiver_x"][::receiver_stride], float(data["array_z"]), float(data["time"][0]),
        float(np.diff(data["time"])[0]), x, z, config.water_speed * speed_factor,
        config.source_center, config.imaging_aperture * aperture_factor, config.maximum_offset)
    return dict(x=x, z=z, migration=values[0], coherent=values[1], coverage=values[2])


def _green(receiver_x, receiver_z, point_x, point_z, frequency, speed):
    """Outgoing 2-D Green function for numpy's exp(-i omega t) transform.

    A real signal uses time dependence exp(+i omega t), hence H_0^(2).
    The normal used in the experiment is downward, nu=(0,-1).
    """
    dx = receiver_x[None, :] - point_x[:, None]
    dz = receiver_z - point_z[:, None]
    radius = np.sqrt(dx ** 2 + dz ** 2)
    k = 2 * np.pi * frequency / speed
    argument = radius[:, :, None] * k
    green = -0.25j * hankel2(0, argument)
    normal_derivative = -0.25j * k * hankel2(1, argument) * (dz / radius)[:, :, None]
    return green, normal_derivative


def dsm_energy(processed, data, config, chunk_size=24, reflection_coefficient=None):
    """Finite-line discretization of Liu (4.4), (5.1), and (5.2).

    The mirror plane is the prescribed reference level config.mean_bottom.
    The positive correlation lags 0 <= tau <= record time are retained.
    Time gating makes this an experimental variant of the original formula.
    Free-water Green functions are used. No acoustic-elastic focusing theorem
    is asserted for this indicator or for the synthetic mirror correction.
    """
    x, z = image_grid(config)
    xx, zz = np.meshgrid(x, z)
    point_x, point_z = xx.ravel(), zz.ravel()
    dt = float(np.diff(data["time"])[0])
    nt = data["pressure"].shape[-1]
    nfft = next_fast_len(2 * nt)
    full_frequency = rfftfreq(nfft, dt)
    selected = np.where((full_frequency >= config.band_low) & (full_frequency <= config.band_high))[0]
    frequency = full_frequency[selected]
    omega = 2 * np.pi * frequency
    phase = np.exp(-1j * omega * float(data["time"][0]))
    pressure_hat = rfft(processed["gated_pressure"], n=nfft, axis=-1)[..., selected] * phase
    derivative_hat = -rfft(processed["gated_gradient"], n=nfft, axis=-1)[..., selected] * phase
    w0 = 2 * np.pi * config.source_frequency
    pulse_hat = config.source_sigma * np.sqrt(2 * np.pi) / 2
    pulse_hat = pulse_hat * (np.exp(-0.5 * config.source_sigma ** 2 * (omega - w0) ** 2)
                             + np.exp(-0.5 * config.source_sigma ** 2 * (omega + w0) ** 2))
    pulse_hat = pulse_hat * np.exp(-1j * omega * config.source_center)
    # The stress-rate source generates (1/c^2) dF/dt in the pressure equation.
    source_hat = 1e4 * 1j * omega / config.water_speed ** 2 * pulse_hat / dt
    receiver_x = data["receiver_x"]
    receiver_weights = line_weights(receiver_x)
    source_weights = line_weights(data["source_x"])
    mirror_z = 2 * config.mean_bottom - float(data["array_z"])
    mirror_source, mirror_source_derivative = _green(
        receiver_x, mirror_z, data["source_x"], np.full_like(data["source_x"], float(data["array_z"])),
        frequency, config.water_speed)
    mirror_source *= source_hat
    mirror_source_derivative *= source_hat
    outputs = {name: np.zeros(len(point_x)) for name in ("dsm", "dsm_corrected", "correction_only")}
    if reflection_coefficient is not None:
        outputs["dsm_impedance"] = np.zeros(len(point_x))
    for start in range(0, len(point_x), chunk_size):
        stop = min(start + chunk_size, len(point_x))
        g, dg = _green(receiver_x, float(data["array_z"]), point_x[start:stop], point_z[start:stop], frequency, config.water_speed)
        g *= source_hat
        dg *= source_hat
        back = np.einsum("srf,qrf,r->sqf", derivative_hat, g.conj(), receiver_weights, optimize=True)
        back -= np.einsum("srf,qrf,r->sqf", pressure_hat, dg.conj(), receiver_weights, optimize=True)
        mg, mdg = _green(receiver_x, mirror_z, point_x[start:stop], point_z[start:stop], frequency, config.water_speed)
        mg *= source_hat
        mdg *= source_hat
        correction = -np.einsum("srf,qrf,r->sqf", mirror_source_derivative, mg.conj(), receiver_weights, optimize=True)
        correction += np.einsum("srf,qrf,r->sqf", mirror_source, mdg.conj(), receiver_weights, optimize=True)
        spectra = [("dsm", back), ("dsm_corrected", back + correction), ("correction_only", correction)]
        if reflection_coefficient is not None:
            # Known normal-incidence half-space reflection coefficient. This is
            # an explicitly approximate candidate for the first-reflection gate.
            spectra.append(("dsm_impedance", back - reflection_coefficient * correction))
        for name, spectrum in spectra:
            full = np.zeros(spectrum.shape[:2] + (len(full_frequency),), dtype=complex)
            full[..., selected] = spectrum
            correlation = irfft(full, n=nfft, axis=-1)[..., :nt]
            outputs[name][start:stop] = dt * np.einsum("sqt,s->q", correlation ** 2, source_weights)
        if start % (chunk_size * 40) == 0:
            print(f"DSM samples {stop}/{len(point_x)}", flush=True)
    return {name: value.reshape(xx.shape) for name, value in outputs.items()}


def estimate_top(processed, data, image, config):
    """Near-zero-offset P-echo height scan with a prescribed [-0.4,0.6] m prior.

    Only the reconstructed bottom, known wave speeds, and measured records enter.
    The formula assumes locally parallel interfaces and a recognizable P echo.
    """
    from .metrics import extract_ridge
    from scipy.signal import hilbert
    bottom, _ = extract_ridge(image["migration"], image["z"])
    candidate_top = np.linspace(-0.4, 0.6, 201)
    profiles, positions, picks = [], [], []
    for shot, sx in enumerate(data["source_x"]):
        if sx < image["x"][0] or sx > image["x"][-1]:
            continue
        receiver = int(np.argmin(abs(data["receiver_x"] - sx)))
        reconstructed_bottom = float(np.interp(sx, image["x"], bottom))
        envelope = abs(hilbert(processed["pressure"][shot, receiver]))
        centre = processed["reference_arrival"][shot, receiver]
        selected = abs(data["time"] - centre) < 0.7 * config.first_window_halfwidth
        indices = np.where(selected)[0]
        bottom_index = indices[np.argmax(envelope[indices])]
        bottom_time = data["time"][bottom_index]
        candidate_time = bottom_time + 2 * (candidate_top - reconstructed_bottom) / config.ice_p_speed
        if candidate_time[-1] >= data["time"][-1]:
            continue
        score = np.interp(candidate_time, data["time"], envelope)
        score /= max(float(score.max()), 1e-30)
        profiles.append(score)
        positions.append(sx)
        picks.append(candidate_top[np.argmax(score)])
    if not profiles:
        return dict(available=False, reason="No usable near-zero-offset traces inside the image aperture or recording time.")
    score = np.mean(profiles, axis=0)
    estimate = float(candidate_top[np.argmax(score)])
    return dict(available=True, candidate_top=candidate_top, score=score,
                trace_scores=np.array(profiles), source_x=np.array(positions),
                individual_top=np.array(picks), estimated_top=estimate,
                individual_std=float(np.std(picks)),
                prior_boundary_peak=bool(np.argmax(score) in (0, len(score) - 1)))


