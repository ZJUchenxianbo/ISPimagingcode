"""Frequency selection and CQ reconstruction for configured experiments."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import ifft

from .quadrature import band_window, quadrature_grid
from .quadrature import benchmark_indices


@dataclass(frozen=True)
class CQGrid:
    carrier_hz: float
    duration_s: float
    dt_s: float
    sigma_per_s: float
    n_out: int
    time_s: np.ndarray
    bins: np.ndarray
    points: np.ndarray
    source: np.ndarray
    indices: np.ndarray
    selected: np.ndarray
    window: np.ndarray

    @classmethod
    def from_config(cls, frequency, band=None, pulse=None):
        """Select FFT carrier offsets in the configured band before Laplace warping."""
        band = frequency if band is None else band
        carrier, duration = frequency["carrier_hz"], frequency["duration_s"]
        low, high = band["min_hz"], band["max_hz"]
        if not np.isfinite([low, high, carrier, duration]).all() or not (
                0 <= low <= carrier <= high and low < high and carrier > 0 and duration > 0):
            raise ValueError("The CQ band must be nonnegative and contain the positive carrier frequency")
        dt = frequency.get("dt_s")
        if dt is None:
            dt = duration / max(3, math.ceil(duration * carrier * 24))
        sigma = frequency.get("sigma_per_s", 4.8 / duration)
        if not np.isfinite([dt, sigma]).all() or dt <= 0 or sigma <= 0:
            raise ValueError("CQ dt_s and sigma_per_s must be positive and finite")
        if high >= 0.5 / dt:
            raise ValueError("The CQ upper frequency must be below the time-grid Nyquist frequency")

        left, right = carrier - low, high - carrier
        tolerance = 8 * np.spacing(high)
        count, time_s, bins, points, source, candidates = quadrature_grid(
            dt, duration, sigma, carrier, max(left, right) + tolerance, pulse=pulse,
        )
        indices = candidates[(bins[candidates] >= -left - tolerance) & (bins[candidates] <= right + tolerance)]
        if "offset_bins" in band:
            requested = band["offset_bins"]
            if ("samples" in band or not isinstance(requested, list) or not requested
                    or any(type(i) is not int for i in requested)):
                raise ValueError("CQ offset_bins must be a nonempty list of integers and cannot be combined with samples")
            if len(set(requested)) != len(requested):
                raise ValueError("CQ offset_bins must contain distinct FFT offsets")
            offsets = np.rint(bins[indices] * len(time_s) * dt).astype(np.intp)
            lookup = dict(zip(offsets, indices))
            outside = sorted(set(requested) - lookup.keys())
            if outside:
                raise ValueError(f"CQ offset_bins outside the available band: {outside}")
            selected = np.array([lookup[i] for i in sorted(requested)], dtype=np.intp)
        else:
            requested = band.get("samples", len(indices))
            if type(requested) is not int or requested < 1:
                raise ValueError("CQ samples must be a positive integer and the band must contain a valid point")
            if requested > len(indices):
                raise ValueError(f"Requested {requested} CQ samples but the band contains {len(indices)}. "
                                 "Reduce samples, widen the band, or increase duration_s.")
            selected = benchmark_indices(indices, bins, requested)
        # Apply the existing cosine taper independently to either side of the carrier.
        relative_shift = bins / np.where(bins < 0, max(left, tolerance), max(right, tolerance))
        window = np.zeros(len(bins))
        window[indices] = band_window(relative_shift[indices], 1.0)
        return cls(carrier, duration, dt, sigma, count, time_s, bins, points, source, indices, selected, window)

    @property
    def bin_offsets(self):
        """Signed integer FFT bin numbers in CQ array order."""
        return np.rint(self.bins * len(self.time_s) * self.dt_s).astype(np.intp)

    @property
    def design_hz(self):
        return float(np.max(self.points[self.indices].imag) / (2 * np.pi))

    def waveform(self, transfer, *, sample_indices=None, output_indices=None, reference_delay_s=None):
        """Reconstruct a real waveform along axis 0, interpolating sparse transfers in-band."""
        transfer = np.asarray(transfer)
        sample_indices = self.selected if sample_indices is None else np.asarray(sample_indices)
        if (sample_indices.ndim != 1 or sample_indices.dtype.kind not in "iu" or not len(sample_indices)
                or len(np.unique(sample_indices)) != len(sample_indices)
                or not np.isin(sample_indices, self.indices).all()
                or transfer.ndim == 0 or transfer.shape[0] != len(sample_indices)):
            raise ValueError("Each transfer row must match a distinct in-band CQ sample index")

        flat = transfer.reshape(len(sample_indices), -1)
        spectrum = np.zeros((len(self.time_s), flat.shape[1]), dtype=complex)
        if len(sample_indices) == len(self.indices):
            spectrum[sample_indices] = flat
        else:
            order = np.argsort(self.bins[sample_indices])
            sampled_bins = self.bins[sample_indices[order]]
            target_bins = self.bins[self.indices]
            if reference_delay_s is not None:
                delay = np.broadcast_to(reference_delay_s, transfer.shape[1:]).reshape(1, -1)
                flat = flat * np.exp(1j * self.points[sample_indices].imag[:, None] * delay)
            for column, values in enumerate(flat.T):
                spectrum[self.indices, column] = np.interp(target_bins, sampled_bins, values[order])
            if reference_delay_s is not None:
                spectrum[self.indices] *= np.exp(-1j * self.points[self.indices].imag[:, None] * delay)
        spectrum *= (self.source * self.window)[:, None]
        samples = slice(self.n_out) if output_indices is None else output_indices
        time = self.time_s[:self.n_out][samples]
        carrier = np.exp((self.sigma_per_s + 2j * np.pi * self.carrier_hz) * time)
        signal = np.real(carrier[:, None] * ifft(spectrum, axis=0, overwrite_x=True)[:self.n_out][samples])
        return signal.reshape((len(time), *transfer.shape[1:]))
