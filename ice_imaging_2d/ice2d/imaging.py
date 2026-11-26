"""First-reflection TD-DSM using the paper finite-time Cauchy pairing."""
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft, rfftfreq
from scipy.special import hankel2
from .geometry import line_weights

def image_grid(config):
    x = np.arange(-config.image_halfwidth, config.image_halfwidth + config.image_x_step / 2, config.image_x_step)
    z = np.arange(config.image_z_min, config.image_z_max + config.image_z_step / 2, config.image_z_step)
    return x, z

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

def dsm_energy(processed, data, config, chunk_size=24):
    """Finite-line discretization of Liu (4.4), (5.1), and (5.2).

    The mirror plane is the prescribed reference level config.mean_bottom.
    The positive correlation lags 0 <= tau <= record time are retained.
    Time gating makes this an experimental variant of the original formula.
    Free-water Green functions are used. No acoustic-elastic focusing theorem
    is asserted for this indicator or for the synthetic mirror correction.

    Kernels are sampled on [0,T), then zero-padded for linear correlation.
    The dt in the Cauchy time integral realizes the finite sum l>=i in (5.1).
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
    if not len(frequency):
        raise ValueError('No Fourier bins in the imaging band')
    if str(data.get('source_equation', '')) != 'wave_equation_rhs':
        raise ValueError('TD-DSM requires explicit wave_equation_rhs source units')
    pressure_hat = rfft(processed["gated_pressure"], n=nfft, axis=-1)
    derivative_hat = -rfft(processed["gated_gradient"], n=nfft, axis=-1)
    if not np.isclose(float(data['time'][0]), 0, atol=1e-14):
        raise ValueError('Finite-time discretization requires records starting at t=0')

    def sampled_kernel(spectrum):
        # Band projection defines Phi_chi; (5.1) then uses only its recorded
        # nonnegative-time samples. Padding after cropping prevents tail wrap.
        full = np.zeros(spectrum.shape[:-1] + (len(full_frequency),), dtype=complex)
        full[..., selected] = spectrum
        trace = irfft(full, n=nfft, axis=-1)[..., :nt]
        return rfft(trace, n=nfft, axis=-1)
    # Phi_chi and measured data use the same effective, filtered source q.
    source_hat = rfft(processed['source_signal'], n=nfft)[selected]
    receiver_x = data["receiver_x"]
    receiver_weights = line_weights(receiver_x)
    source_weights = line_weights(data["source_x"])
    mirror_z = 2 * config.mean_bottom - float(data["array_z"])
    mirror_source, mirror_source_derivative = _green(
        receiver_x, mirror_z, data["source_x"], np.full_like(data["source_x"], float(data["array_z"])),
        frequency, config.water_speed)
    mirror_source *= source_hat
    mirror_source_derivative *= source_hat
    if 'receiver_mask' in processed:
        # The same source-dependent finite aperture must enter V^s and V_corr^i.
        mask = np.asarray(processed['receiver_mask'])
        if mask.shape != pressure_hat.shape[:2]:
            raise ValueError('receiver_mask must have shape (source, receiver)')
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError('receiver_mask must contain finite weights in [0, 1]')
        pressure_hat *= mask[..., None]
        derivative_hat *= mask[..., None]
        mirror_source *= mask[..., None]
        mirror_source_derivative *= mask[..., None]
    mirror_source = sampled_kernel(mirror_source)
    mirror_source_derivative = sampled_kernel(mirror_source_derivative)
    outputs = {name: np.zeros(len(point_x)) for name in ("dsm", "dsm_corrected", "correction_only")}
    for start in range(0, len(point_x), chunk_size):
        stop = min(start + chunk_size, len(point_x))
        g, dg = _green(receiver_x, float(data["array_z"]), point_x[start:stop], point_z[start:stop], frequency, config.water_speed)
        g *= source_hat
        dg *= source_hat
        g, dg = sampled_kernel(g), sampled_kernel(dg)
        back = np.einsum("srf,qrf,r->sqf", derivative_hat, g.conj(), receiver_weights, optimize=True)
        back -= np.einsum("srf,qrf,r->sqf", pressure_hat, dg.conj(), receiver_weights, optimize=True)
        mg, mdg = _green(receiver_x, mirror_z, point_x[start:stop], point_z[start:stop], frequency, config.water_speed)
        mg *= source_hat
        mdg *= source_hat
        mg, mdg = sampled_kernel(mg), sampled_kernel(mdg)
        correction = -np.einsum("srf,qrf,r->sqf", mirror_source_derivative, mg.conj(), receiver_weights, optimize=True)
        correction += np.einsum("srf,qrf,r->sqf", mirror_source, mdg.conj(), receiver_weights, optimize=True)
        spectra = [("dsm", back), ("dsm_corrected", back + correction), ("correction_only", correction)]
        for name, spectrum in spectra:
            correlation = dt * irfft(spectrum, n=nfft, axis=-1)[..., :nt]
            outputs[name][start:stop] = dt * np.einsum("sqt,s->q", correlation ** 2, source_weights)
        if start % (chunk_size * 40) == 0:
            print(f"DSM samples {stop}/{len(point_x)}", flush=True)
    return {name: value.reshape(xx.shape) for name, value in outputs.items()}
