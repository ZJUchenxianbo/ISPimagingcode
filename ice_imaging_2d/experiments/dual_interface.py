"""Three experiment families from the proposal; all algorithms live in ice2d."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import platform
import time
import numpy as np

from ..ice2d.geometry import Geometry
from ..ice2d.model import ice_bottom
from ..ice2d.solver import SOLVER_VERSION, simulate
from ..ice2d.sensitivity import CoupledForward, WaveformObjective
from ..ice2d.reconstruction import reconstruct
from ..ice2d.dual_outputs import save_dual_results
from ..ice2d.processing import process_records
from ..ice2d.imaging import migrate, dsm_energy, estimate_top
from ..ice2d.metrics import evaluate
from ..ice2d.data_store import identifier, write_manifest
from ..ice2d.sensitivity import linearize, sensitivity_diagnostics


def run_dual_experiment(physics, inverse, experiment, name, output, workers=1,
                        reuse_data=False, skip_dsm=False, stage="reconstruct", strategy="frozen", model_budget=18, shared_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shared = Path(shared_root) if shared_root else output
    observation_settings = asdict(experiment)
    observation_settings.pop("baselines")
    observation_settings.pop("parameter_notes")
    dataset_signature = dict(physics=asdict(physics), experiment=observation_settings, name=name,
                             control_x=inverse.control_x, solver_version=SOLVER_VERSION)
    dataset = shared / "datasets" / identifier(dataset_signature)
    dataset.mkdir(parents=True, exist_ok=True)
    signature = dict(stage=stage, strategy=strategy, model_budget=model_budget,physics=asdict(physics), inverse=asdict(inverse), experiment=asdict(experiment),
                     name=name, solver_version=SOLVER_VERSION)
    # JSON roundtrip normalizes tuple-valued nodes.
    signature = json.loads(json.dumps(signature))
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != signature:
        raise ValueError(f"{output} belongs to a different model/inversion configuration")
    config_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2))
    started = time.perf_counter()
    mode = "bottom" if name.startswith("bottom") else name
    x_truth = np.linspace(inverse.control_x[0], inverse.control_x[-1], 257)
    shape = "flat" if mode == "top" else "bump" if name == "bottom_bump" else "sine"
    top = physics.ice_top + (experiment.top_shift if mode in ("top", "joint") else 0)
    truth = Geometry(top, tuple(x_truth), tuple(ice_bottom(x_truth, physics, shape)))
    truth_physics = replace(physics, grid_spacing=physics.grid_spacing / experiment.truth_grid_ratio).validate()
    truth.validate(truth_physics)
    forward = CoupledForward(physics, inverse, shared / "forward_cache", workers, quiet=False)
    print(f"[{name}] Generating reference data", flush=True)
    reference = forward.evaluate(np.zeros(forward.parameter_count))
    np.savez_compressed(dataset / "data_reference.npz", **reference)
    observed_path = dataset / "data_observed.npz"
    if reuse_data or observed_path.exists():
        if not observed_path.exists():
            raise ValueError("--reuse-data requires data_observed.npz in the same experiment directory")
        from ..ice2d.io import load_data
        observations = load_data(observed_path)
        if str(observations["solver_version"]) != SOLVER_VERSION:
            raise ValueError("Incompatible solver version in observed data")
        if json.loads(str(observations["geometry_json"])) != json.loads(json.dumps(truth.as_dict())):
            raise ValueError("Incompatible observed geometry")
        if json.loads(str(observations["config_json"])) != truth_physics.as_dict():
            raise ValueError("Incompatible observation physics/grid")
    else:
        print(f"[{name}] Generating observed data", flush=True)
        observations = simulate(truth_physics, "flat", shared / "forward_cache", workers,
                                geometry=truth, save=False, quiet=False)
        # Optional synthetic pressure noise, scaled globally before inversion.
        # It is not the legacy residual-window SNR convention.
        if experiment.noise_snr_db is not None:
            rng = np.random.default_rng(physics.noise_seed)
            noise = rng.normal(size=observations["pressure"].shape)
            noise *= (np.linalg.norm(observations["pressure"]) / np.linalg.norm(noise)
                      * 10 ** (-experiment.noise_snr_db / 20))
            observations["pressure"] = observations["pressure"] + noise
        np.savez_compressed(observed_path, **observations)
    write_manifest(dataset / "metadata.json", dataset_signature)
    write_manifest(output / "data_manifest.json", dict(dataset=str(dataset.resolve()), stage=stage))
    from ..ice2d.data_figures import save_data_figures
    save_data_figures(dataset, truth_physics, observations, truth)
    if stage == "data":
        print(f"Data ready: {dataset}", flush=True)
        return dict(status="data_ready")
    objective = WaveformObjective(physics, inverse, reference, observations)
    print(f"[{name}] Computing interface sensitivities and reconstruction", flush=True)
    if stage == "indicator":
        columns = linearize(forward, objective, np.zeros(forward.parameter_count), output / "cache")
        arrays = {}
        for window in ("bottom", "internal", "full"):
            for key, value in sensitivity_diagnostics(columns, objective.initial_residual, objective, window).items():
                arrays[f"{window}_{key}"] = value
        np.savez_compressed(output / "interface_indicators.npz", **arrays)
        return dict(status="indicator_ready")
    result = reconstruct(forward, objective, output, mode, strategy, model_budget)
    # The truth enters only this evaluation/output layer, never reconstruct().
    baseline_metrics = {}
    if experiment.baselines:
        water = simulate(truth_physics, "background", shared / "forward_cache", workers,
                         save=False, quiet=True)
        processed = process_records(observations, water, truth_physics)
        baseline = migrate(processed, observations, truth_physics)
        if not skip_dsm and experiment.noise_snr_db is None:
            impedance_w = physics.water_density * physics.water_speed
            impedance_i = physics.ice_density * physics.ice_p_speed
            baseline.update(dsm_energy(processed, observations, truth_physics,
                                      reflection_coefficient=(impedance_i - impedance_w) / (impedance_i + impedance_w)))
        for method in ("migration", "coherent", "dsm", "dsm_corrected", "dsm_impedance"):
            if method in baseline:
                baseline_metrics[method] = evaluate(baseline[method], baseline["x"], baseline["z"],
                                                    truth.evaluate(baseline["x"]), inverse.reference_bottom)
        top_scan = estimate_top(processed, observations, baseline, truth_physics)
        if top_scan["available"]:
            np.savez_compressed(output / "baseline_top_scan.npz", **top_scan)
            baseline_metrics["top_scan"] = dict(estimated_top=top_scan["estimated_top"],
                                                absolute_error_m=abs(top_scan["estimated_top"] - truth.top))
        np.savez_compressed(output / "baseline_images.npz", **baseline)
    import numba
    import scipy
    metadata = dict(**signature, truth_geometry=truth.as_dict(), reference_geometry=forward.geometry(np.zeros(forward.parameter_count)).as_dict(),
                    diagnostic_geometry="reference" if strategy == "frozen" else "final",
                    method="central-difference coupled-interface sensitivity + regularized Gauss-Newton",
                    continuous_shape_derivative=False, adjoint_rtm=False,
                    source_x=reference["source_x"].tolist(), receiver_x=reference["receiver_x"].tolist(),
                    array_z=float(reference["array_z"]), record_shape=list(observations["pressure"].shape),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                    workers=workers, reused_data=reuse_data, dsm_baseline_computed="dsm" in baseline_metrics,
                    objective_normalization=objective.scale,
                    elapsed_seconds=time.perf_counter() - started)
    save_dual_results(output, inverse, objective, result, truth, metadata, baseline_metrics)
    print(f"Completed {name}: {output}, status={result['status']}", flush=True)
    return result
