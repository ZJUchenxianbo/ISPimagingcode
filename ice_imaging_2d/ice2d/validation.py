"""Physical checks that can detect an incorrect forward model or source convention."""

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.special import hankel2
from scipy.fft import rfftfreq, irfft, next_fast_len

from .solver import simulate
from .processing import process_records


def _peak(time, signal, predicted, half_width=0.00015):
    indices = np.where(abs(time - predicted) < half_width)[0]
    envelope = abs(hilbert(signal))
    return float(time[indices[np.argmax(envelope[indices])]])


def physical_checks(config, flat, background):
    processed = process_records(flat, background, config)
    s = len(flat["source_x"]) // 2
    r = int(np.argmin(abs(flat["receiver_x"] - flat["source_x"][s])))
    tb = config.source_center + 2 * (config.mean_bottom - float(flat["array_z"])) / config.water_speed
    ta = tb + 2 * (config.ice_top - config.mean_bottom) / config.ice_p_speed
    observed_bottom = _peak(flat["time"], processed["pressure"][s, r], tb)
    observed_top = _peak(flat["time"], processed["pressure"][s, r], ta)
    # Homogeneous 2-D Hankel Green function checks source normalization and phase.
    r_direct = int(np.argmin(abs(background["receiver_x"] - background["source_x"][s] - 0.8)))
    radius = abs(background["receiver_x"][r_direct] - background["source_x"][s])
    dt = float(np.diff(background["time"])[0])
    nfft = next_fast_len(4 * len(background["time"]))
    frequency = rfftfreq(nfft, dt)
    omega = 2 * np.pi * frequency[1:]
    w0 = 2 * np.pi * config.source_frequency
    pulse_hat = config.source_sigma * np.sqrt(2 * np.pi) / 2
    pulse_hat = pulse_hat * (np.exp(-0.5 * config.source_sigma ** 2 * (omega - w0) ** 2)
                             + np.exp(-0.5 * config.source_sigma ** 2 * (omega + w0) ** 2))
    source_hat = pulse_hat * np.exp(-1j * omega * config.source_center) * 1e4 * 1j * omega / config.water_speed ** 2 / dt
    spectrum = np.zeros(len(frequency), dtype=complex)
    spectrum[1:] = source_hat * (-0.25j * hankel2(0, omega * radius / config.water_speed))
    spectrum[1:] *= np.exp(1j * omega * float(background["time"][0]))
    analytic = irfft(spectrum, n=nfft)[:len(background["time"])]
    sos = butter(4, [config.band_low, config.band_high], fs=1 / dt, btype="bandpass", output="sos")
    analytic = sosfiltfilt(sos, analytic)
    numerical = sosfiltfilt(sos, background["pressure"][s, r_direct])
    direct_window = abs(background["time"] - config.source_center - radius / config.water_speed) < 0.00018
    waveform_error = float(np.linalg.norm(numerical[direct_window] - analytic[direct_window]) / np.linalg.norm(analytic[direct_window]))
    reciprocity_errors = []
    for a in range(len(flat["source_x"])):
        ra = int(np.argmin(abs(flat["receiver_x"] - flat["source_x"][a])))
        if abs(flat["receiver_x"][ra] - flat["source_x"][a]) > config.grid_spacing / 3:
            continue
        for b in range(a + 1, len(flat["source_x"])):
            rb = int(np.argmin(abs(flat["receiver_x"] - flat["source_x"][b])))
            if abs(flat["receiver_x"][rb] - flat["source_x"][b]) > config.grid_spacing / 3:
                continue
            first, second = flat["pressure"][a, rb], flat["pressure"][b, ra]
            reciprocity_errors.append(float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-30)))
    result = dict(predicted_bottom_ms=tb * 1000, measured_bottom_ms=observed_bottom * 1000,
                  predicted_top_ms=ta * 1000, measured_top_ms=observed_top * 1000,
                  echo_separation_error_us=((observed_top - observed_bottom) - (ta - tb)) * 1e6,
                  direct_wave_relative_l2_error=waveform_error,
                  reciprocal_pairs=len(reciprocity_errors),
                  maximum_reciprocity_error=max(reciprocity_errors) if reciprocity_errors else None)
    result["arrival_check_pass"] = bool(abs(result["echo_separation_error_us"]) < 25)
    result["direct_wave_check_pass"] = waveform_error < 0.1
    return result


def refinement_check(config, output, workers=1):
    directory = Path(output) / "refinement"
    # The nearest cell centres differ by a known subcell amount between grids.
    # Correct only that acquisition-position shift when comparing arrivals.
    coarse = replace(config, source_count=1, source_halfwidth=0,
                     record_seconds=min(config.record_seconds, 0.0035))
    fine = replace(coarse, grid_spacing=config.grid_spacing / 2)
    records = []
    for label, setup in (("coarse", coarse), ("fine", fine)):
        data = simulate(setup, "flat", directory / label, workers)
        background = simulate(setup, "background", directory / label, workers)
        processed = process_records(data, background, setup)
        receiver = int(np.argmin(abs(data["receiver_x"] - data["source_x"][0])))
        t0 = setup.source_center + 2 * (setup.mean_bottom - float(data["array_z"])) / setup.water_speed
        arrival = _peak(data["time"], processed["pressure"][0, receiver], t0)
        records.append((data, processed["pressure"][0, receiver], arrival, t0))
    coarse_data, coarse_trace, coarse_arrival, coarse_prediction = records[0]
    fine_data, fine_trace, fine_arrival, fine_prediction = records[1]
    # Remove only the known subcell acquisition-position difference.
    fine_on_coarse = np.interp(coarse_data["time"] + fine_prediction - coarse_prediction, fine_data["time"], fine_trace)
    window = abs(coarse_data["time"] - coarse_prediction) < 0.00020
    relative_error = float(np.linalg.norm(coarse_trace[window] - fine_on_coarse[window]) / np.linalg.norm(fine_on_coarse[window]))
    difference = ((coarse_arrival - coarse_prediction) - (fine_arrival - fine_prediction)) * 1e6
    result = dict(coarse_spacing_m=coarse.grid_spacing, fine_spacing_m=fine.grid_spacing,
                  first_echo_relative_l2_difference=relative_error,
                  position_corrected_arrival_difference_us=difference,
                  scope="One flat-model near-zero-offset first ice-bottom echo.")
    (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(directory / "comparison.npz", time=coarse_data["time"],
                        coarse=coarse_trace, fine_resampled=fine_on_coarse, window=window)
    return result


def reciprocity_check(config, output, workers=1):
    """Use exactly collocated source/receiver pairs for a meaningful swap test."""
    setup = replace(config, source_count=2, source_halfwidth=0.6,
                    receiver_halfwidth=0.6, receiver_spacing=1.2,
                    record_seconds=min(config.record_seconds, 0.0035))
    data = simulate(setup, "flat", Path(output) / "reciprocity", workers)
    if not np.array_equal(data["source_x"], data["receiver_x"]):
        raise RuntimeError("The dedicated reciprocity acquisition did not collocate.")
    first, second = data["pressure"][0, 1], data["pressure"][1, 0]
    difference = float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-30))
    return dict(relative_l2_error=difference, passed=difference < 1e-3,
                source_x_m=data["source_x"].tolist(), scope="Dedicated two-shot flat-model reciprocal acquisition.")

