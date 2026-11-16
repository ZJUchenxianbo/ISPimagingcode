"""Background subtraction, reproducible noise, and observable-data time windows."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def process_records(data, background, config, snr_db=None, speed_factor=1.0):
    for name in ("time", "source_x", "receiver_x", "array_z"):
        if not np.array_equal(data[name], background[name]):
            raise ValueError(f"Incompatible background {name}.")
    dt = float(np.diff(data["time"])[0])
    sos = butter(4, [config.band_low, config.band_high], fs=1 / dt, btype="bandpass", output="sos")
    pressure = sosfiltfilt(sos, data["pressure"] - background["pressure"], axis=-1)
    gradient = sosfiltfilt(sos, data["gradient"] - background["gradient"], axis=-1)
    source_x, receiver_x = data["source_x"], data["receiver_x"]
    offset = receiver_x[None, :] - source_x[:, None]
    height = config.mean_bottom - float(data["array_z"])
    reference_arrival = config.source_center + np.sqrt(offset ** 2 + (2 * height) ** 2) / (config.water_speed * speed_factor)
    difference = data["time"][None, None, :] - reference_arrival[:, :, None]
    half = config.first_window_halfwidth
    # Flat central window with a cosine rolloff near its outer quarter.
    rolloff = np.clip((abs(difference) - 0.75 * half) / (0.25 * half), 0, 1)
    window = 0.5 * (1 + np.cos(np.pi * rolloff))
    noise_info = {}
    if snr_db is not None:
        rng = np.random.default_rng(config.noise_seed)
        for name, signal in (("pressure", pressure), ("gradient", gradient)):
            noise = sosfiltfilt(sos, rng.normal(size=signal.shape), axis=-1)
            scale = np.linalg.norm(signal * window) / np.linalg.norm(noise * window)
            noise *= scale * 10 ** (-snr_db / 20)
            noise_info[name] = float(20 * np.log10(np.linalg.norm(signal * window) / np.linalg.norm(noise * window)))
            signal += noise
    return dict(pressure=pressure, gradient=gradient,
                gated_pressure=pressure * window, gated_gradient=gradient * window,
                analytic=hilbert(pressure, axis=-1) * window,
                reference_arrival=reference_arrival, window=window,
                snr_db=snr_db, measured_snr=noise_info)

