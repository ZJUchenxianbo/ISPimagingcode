"""Small numerical acceptance checks, independent of reconstruction truth fitting."""

from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.signal import hilbert

from .config import Config
from .geometry import Geometry, sampling_stencil
from .model import make_model
from .solver import _advance, simulate
from .sensitivity import CoupledForward, WaveformObjective, linearize, sensitivity_diagnostics, resample_pressure
from .reconstruction import reconstruct


def plane_layer_checks():
    """Normal-incidence plane pulse: exact scalar impedances apply only here.

    A wide line source gives a plane wave in the central causal region. Compare
    the separated bottom and first top echoes to shifted homogeneous records.
    No scalar impedance claim is made for oblique P/S conversion.
    """
    config = replace(Config(), upper_medium="air", interface_scheme="smooth", acquisition="linear",
                     snapshots=False, spatial_order=2, horizontal_halfwidth=5.0, computational_top=0.6,
                     seabed_depth=3.0, array_depth=2.0, source_count=1, source_halfwidth=0,
                     receiver_halfwidth=0.1, receiver_spacing=0.1, grid_spacing=0.025,
                     sponge_width=0.3, source_frequency=1200., band_low=500., band_high=2000.,
                     source_sigma=0.00012, source_center=0.0008, record_seconds=0.0032,
                     output_interval=0.00001, rough_amplitude=0).validate()

    def plane(cfg, case):
        model = make_model(cfg, case)
        h = cfg.grid_spacing
        source_x = model["x"][3:-3]
        source_nodes, weights = sampling_stencil(model["x"], model["z"], source_x, 2.0, "linear")
        receiver_nodes, receiver_weights = sampling_stencil(model["x"], model["z"], [0.0], 1.9, "linear")
        times = (np.arange(model["steps"]) + 0.5) * model["dt"]
        pulse = np.exp(-0.5 * ((times - cfg.source_center) / cfg.source_sigma) ** 2)
        pulse *= np.cos(2 * np.pi * cfg.source_frequency * (times - cfg.source_center))
        pressure, _, _ = _advance(model["lam"], model["mu"], model["corner_mu"],
                                  model["inverse_density_x"], model["inverse_density_z"],
                                  model["damping"], model["dt"], h, (pulse * 1e4 / h).astype(np.float32),
                                  source_nodes.reshape(-1, 2), weights.ravel(), receiver_nodes,
                                  receiver_weights, model["stride"], np.empty(0, dtype=np.int64), cfg.spatial_order)
        time = (np.arange(pressure.shape[-1]) * model["stride"] + 1) * model["dt"]
        if not np.isfinite(pressure).all():
            raise AssertionError("Plane-wave run is unstable")
        return time, pressure[0]

    time, homogeneous = plane(config, "background")
    _, layer = plane(config, "flat")
    reflection = layer - homogeneous
    water_impedance = config.water_density * config.water_speed
    ice_impedance = config.ice_density * config.ice_p_speed
    air_impedance = config.air_density * config.air_speed
    bottom_exact = (ice_impedance - water_impedance) / (ice_impedance + water_impedance)
    top_exact = (4 * water_impedance * ice_impedance / (water_impedance + ice_impedance) ** 2
                 * (air_impedance - ice_impedance) / (air_impedance + ice_impedance))
    direct_arrival = config.source_center + 0.1 / config.water_speed
    delay_bottom = 2 * (1.9 + config.mean_bottom) / config.water_speed
    delay_top = delay_bottom + 2 * (config.ice_top - config.mean_bottom) / config.ice_p_speed
    estimates = []
    def envelope_arrival(signal, predicted):
        envelope = np.abs(hilbert(signal))
        selected = np.flatnonzero(abs(time - predicted) < 1.5 * config.source_sigma)
        return float(time[selected[np.argmax(envelope[selected])]])

    direct_error = abs(envelope_arrival(homogeneous, direct_arrival) - direct_arrival)
    bottom_error = abs(envelope_arrival(reflection, direct_arrival + delay_bottom)
                       - direct_arrival - delay_bottom)
    top_error = abs(envelope_arrival(reflection, direct_arrival + delay_top)
                    - direct_arrival - delay_top)
    for delay in (delay_bottom, delay_top):
        shifted = np.interp(time - delay, time, homogeneous, left=0, right=0)
        mask = abs(time - direct_arrival - delay) <= 1.5 * config.source_sigma
        estimates.append(float(np.dot(reflection[mask], shifted[mask]) / np.dot(shifted[mask], shifted[mask])))
    # Convergence / air-loading comparisons use identical physical observation points.
    fine = replace(config, grid_spacing=config.grid_spacing / 2)
    fine_time, fine_layer = plane(fine, "flat")
    aligned = np.interp(time, fine_time, fine_layer)
    window = (time > direct_arrival + delay_bottom - 2 * config.source_sigma)
    window &= time < direct_arrival + delay_top + 2 * config.source_sigma
    refinement = float(np.linalg.norm(layer[window] - aligned[window]) / np.linalg.norm(aligned[window]))
    _, free = plane(replace(config, upper_medium="free_surface"), "flat")
    _, light_air = plane(replace(config, air_density=config.air_density * 1e-3), "flat")
    free_difference = float(np.linalg.norm(layer[window] - free[window]) / np.linalg.norm(free[window]))
    limit_difference = float(np.linalg.norm(light_air[window] - free[window]) / np.linalg.norm(free[window]))
    return dict(bottom_reflection_exact=bottom_exact, bottom_reflection_measured=estimates[0],
                top_echo_exact=top_exact, top_echo_measured=estimates[1],
                bottom_amplitude_absolute_error=abs(estimates[0] - bottom_exact),
                top_amplitude_absolute_error=abs(estimates[1] - top_exact),
                refinement_relative_difference=refinement,
                air_vs_free_surface=free_difference, light_air_vs_free_surface=limit_difference,
                direct_arrival_error_seconds=direct_error, bottom_arrival_error_seconds=bottom_error,
                top_arrival_error_seconds=top_error,
                passed=bool(abs(estimates[0] - bottom_exact) < 0.05 and abs(estimates[1] - top_exact) < 0.05
                            and refinement < 0.2 and limit_difference < max(free_difference, 0.005)
                            and max(direct_error, bottom_error, top_error) < 0.00008),
                scope="1-D normal-incidence causal region in a wide 2-D model; empirical regression tolerances, not full convergence")


def sensitivity_checks(physics, inverse, directory):
    directory = Path(directory)
    forward = CoupledForward(physics, inverse, directory / "forward")
    theta = np.zeros(forward.parameter_count)
    reference = forward.evaluate(theta)
    # An independent, deterministic direction probes all parameters.
    direction = np.cos(np.arange(forward.parameter_count) + 0.3)
    direction /= np.linalg.norm(direction)
    observations = forward.evaluate(0.003 * direction)
    objective = WaveformObjective(physics, inverse, reference, observations)
    columns = linearize(forward, objective, theta, directory / "standard")
    smaller = linearize(forward, objective, theta, directory / "half_step", step=inverse.difference_step / 2)
    weights = objective.weight().ravel()

    def norm(values):
        return np.sqrt(np.sum(values.ravel() ** 2 * weights))

    step_error = np.sqrt(sum(norm(a - b) ** 2 for a, b in zip(columns, smaller))) / max(np.sqrt(sum(norm(row) ** 2 for row in smaller)), 1e-30)
    predicted_derivative = direction @ columns
    errors = []
    for epsilon in (0.004, 0.002):
        data = forward.evaluate(epsilon * direction)
        actual = objective.filter(data["pressure"]) - objective.filter(reference["pressure"])
        errors.append(float(norm(actual.ravel() - epsilon * predicted_derivative) / max(norm(actual), 1e-30)))
    diagnostic = sensitivity_diagnostics(columns, objective.initial_residual, objective)
    # Change only labels/metadata used for evaluation. Inversion input processing
    # must produce exactly the same objective; no truth geometry is consulted.
    changed = dict(observations, geometry_json=np.array('{"top":999,"bottom":[999]}'))
    alternate = WaveformObjective(physics, inverse, reference, changed)
    truth_isolation = np.array_equal(objective.observed, alternate.observed)
    # Distinct forward grids: compare filtered directional responses on common times.
    fine_physics = replace(physics, grid_spacing=physics.grid_spacing / 1.25)
    fine_forward = CoupledForward(fine_physics, inverse, directory / "fine")
    step = inverse.difference_step
    fine_plus, fine_minus = fine_forward.evaluate(step * direction), fine_forward.evaluate(-step * direction)
    plus, valid_plus = resample_pressure(fine_plus, reference)
    minus, valid_minus = resample_pressure(fine_minus, reference)
    fine_derivative = objective.filter((plus - minus) / (2 * step)).ravel()
    valid_weight = weights * (valid_plus & valid_minus)[None, None, :].repeat(len(reference["source_x"]), axis=0).repeat(len(reference["receiver_x"]), axis=1).ravel()
    grid_difference = float(np.sqrt(np.sum((predicted_derivative - fine_derivative) ** 2 * valid_weight))
                            / max(np.sqrt(np.sum(fine_derivative ** 2 * valid_weight)), 1e-30))
    del columns, smaller
    return dict(difference_step_relative_change=float(step_error), linearization_relative_errors=errors,
                fine_grid_directional_difference=grid_difference,
                minimum_column_norm=float(diagnostic["column_norms"].min()),
                full_parameter_rank=diagnostic["rank"], truth_metadata_isolation=bool(truth_isolation),
                passed=bool(step_error < 0.15 and errors[1] < 0.15 and errors[1] < errors[0]
                            and diagnostic["column_norms"].min() > 0 and truth_isolation and grid_difference < 0.2),
                scope="Small low-frequency discrete linearization; grid tolerance detects gross failure, not formal convergence")


def acquisition_checks(physics, directory):
    physics = replace(physics, source_count=2, source_halfwidth=0.25,
                      receiver_halfwidth=0.25, receiver_spacing=0.5)
    serial = simulate(physics, "flat", Path(directory) / "serial", 1, quiet=True, save=False)
    parallel = simulate(physics, "flat", Path(directory) / "parallel", 2, quiet=True, save=False)
    cached = simulate(physics, "flat", Path(directory) / "parallel", 2, quiet=True, save=False)
    first, second = serial["pressure"][0, 1], serial["pressure"][1, 0]
    reciprocity = float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-30))
    consistent = np.array_equal(serial["pressure"], parallel["pressure"]) and np.array_equal(parallel["pressure"], cached["pressure"])
    stale = dict(serial, solver_version=np.array("staggered-4-2-v1"))
    rejected = False
    try:
        resample_pressure(stale, serial)
    except ValueError:
        rejected = True
    return dict(reciprocity_relative_error=reciprocity, serial_parallel_cache_equal=bool(consistent),
                old_data_rejected=rejected, passed=bool(reciprocity < 1e-4 and consistent and rejected))


def local_reconstruction_check(physics, inverse, directory):
    """A small representable perturbation checks the local inverse separately
    from geometry approximation and grid mismatch in the public experiments.
    """
    directory = Path(directory) / "local_inverse"
    forward = CoupledForward(physics, inverse, directory / "forward")
    theta = 0.001 * np.cos(np.arange(forward.parameter_count) * 0.6)
    observations = forward.evaluate(theta)
    reference = forward.evaluate(np.zeros(forward.parameter_count))
    objective = WaveformObjective(physics, inverse, reference, observations)
    result = reconstruct(forward, objective, directory / "original")
    labeled = dict(observations, geometry_json=np.array('{"top":999,"bottom":[999]}'))
    isolated_objective = WaveformObjective(physics, inverse, reference, labeled)
    isolated = reconstruct(forward, isolated_objective, directory / "relabeled")
    error = float(np.linalg.norm(result["theta"] - theta) / np.linalg.norm(theta))
    invariant = np.array_equal(result["theta"], isolated["theta"])
    return dict(relative_parameter_error=error, final_relative_residual=result["final_norm"],
                truth_metadata_invariant=bool(invariant),
                passed=bool(error < 0.2 and result["final_norm"] < 0.1 and invariant),
                scope="Small representable same-grid perturbation; independent-grid experiments are separate")
