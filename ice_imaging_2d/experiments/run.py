"""Experiment orchestration, machine-readable outputs, and a Chinese report."""

import csv
import json
import platform
import time
from pathlib import Path
import numpy as np

from ..ice2d.io import load_data
from ..ice2d.config import Config
from ..ice2d.reporting import write_report
from ..ice2d.validation import physical_checks, refinement_check, reciprocity_check
from ..ice2d.solver import simulate
from ..ice2d.model import ice_bottom
from ..ice2d.processing import process_records
from ..ice2d.imaging import migrate, dsm_energy, estimate_top
from ..ice2d.metrics import evaluate, extract_ridge
from ..ice2d.plotting import (model_and_pulse, gather_plot, inversion_plot, dsm_plot,
                       snapshot_plot, robustness_plot)


def run_case(config, case, output, workers=1, with_dsm=True, reuse_data=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if reuse_data:
        background = load_data(output / "data_background.npz")
        data = load_data(output / f"data_{case}.npz")
        for item in (background, data):
            if Config(**json.loads(str(item["config_json"]))).as_dict() != config.as_dict():
                raise ValueError("Saved data configuration differs from the requested configuration.")
    else:
        background = simulate(config, "background", output, workers)
        data = simulate(config, case, output, workers)
    processed = process_records(data, background, config)
    result = migrate(processed, data, config)
    if with_dsm:
        water_impedance = config.water_density * config.water_speed
        ice_impedance = config.ice_density * config.ice_p_speed
        reflection = (ice_impedance - water_impedance) / (ice_impedance + water_impedance)
        if case == "soft":
            reflection = -1.0
        result.update(dsm_energy(processed, data, config, reflection_coefficient=reflection))
    truth = ice_bottom(result["x"], config, case)
    methods = [name for name in ("migration", "coherent", "dsm", "dsm_corrected", "dsm_impedance", "correction_only") if name in result]
    metrics = {name: evaluate(result[name], result["x"], result["z"], truth, config.mean_bottom) for name in methods}
    np.savez_compressed(output / f"images_{case}.npz", **result, true_bottom=truth)
    (output / f"metrics_{case}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / f"profile_{case}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "true_bottom_m"] + [f"{name}_bottom_m" for name in methods])
        ridges = [extract_ridge(result[name], result["z"])[0] for name in methods]
        writer.writerows(zip(result["x"], truth, *ridges))
    model_and_pulse(data, config, case, output)
    gather_plot(data, background, processed, config, case, output)
    inversion_plot(result, config, case, output)
    dsm_plot(result, config, case, output)
    snapshot_plot(data, config, case, output)
    if case != "soft":
        top = estimate_top(processed, data, result, config)
        if top["available"]:
            import matplotlib.pyplot as plt
            from ..ice2d.plotting import save_figure
            np.savez_compressed(output / f"top_scan_{case}.npz", **top)
            top_summary = {key: value for key, value in top.items() if not isinstance(value, np.ndarray)}
            top_summary["true_top_m"] = config.ice_top
            top_summary["absolute_error_m"] = abs(top["estimated_top"] - config.ice_top)
            (output / f"top_metrics_{case}.json").write_text(json.dumps(top_summary, indent=2), encoding="utf-8")
            fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
            for score in top["trace_scores"]:
                ax.plot(top["candidate_top"], score, color="#aaaaaa", alpha=0.45, lw=0.8)
            ax.plot(top["candidate_top"], top["score"], color="#315f7e", lw=2, label="Mean echo envelope")
            ax.axvline(config.ice_top, color="black", ls="--", label="True top")
            ax.axvline(top["estimated_top"], color="#c56840", ls=":", label="Estimated top")
            ax.set(xlabel="Candidate flat ice-top z (m)", ylabel="Normalized envelope", title="P-echo travel-time height scan")
            ax.legend(fontsize=8)
            save_figure(fig, output, f"{case}_top_scan")
    print(json.dumps({case: metrics}, indent=2), flush=True)
    return data, background, result, metrics


def run_robustness(config, data, background, output):
    settings = [
        ("Clean", None, 1.0, 1.0, 1),
        ("20 dB noise", 20.0, 1.0, 1.0, 1),
        ("10 dB noise", 10.0, 1.0, 1.0, 1),
        ("Water speed -2%", None, 0.98, 1.0, 1),
        ("Water speed +2%", None, 1.02, 1.0, 1),
        ("Half local aperture", None, 1.0, 0.5, 1),
        ("Every fourth receiver", None, 1.0, 1.0, 4),
    ]
    rows = []
    saved = {}
    for name, snr, speed, aperture, stride in settings:
        # The reference window is fixed before these perturbations.
        processed = process_records(data, background, config, snr_db=snr)
        result = migrate(processed, data, config, speed_factor=speed, aperture_factor=aperture, receiver_stride=stride)
        truth = ice_bottom(result["x"], config, "sine")
        metrics = evaluate(result["migration"], result["x"], result["z"], truth, config.mean_bottom)
        coherent_metrics = evaluate(result["coherent"], result["x"], result["z"], truth, config.mean_bottom)
        rows.append(dict(experiment=name, **metrics, coherent_rmse_m=coherent_metrics["rmse_m"],
                         coherent_peak_to_valley_m=coherent_metrics["peak_to_valley_m"],
                         requested_snr_db=snr, measured_snr=processed["measured_snr"]))
        saved[name] = result["migration"]
        saved[name + " coherent"] = result["coherent"]
    np.savez_compressed(Path(output) / "robustness_images.npz", **saved, x=result["x"], z=result["z"])
    (Path(output) / "robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    robustness_plot(rows, output)
    return rows


def run_experiments(config, case, args):
    """按入口参数组织算例，保留原实验顺序和科学参数。"""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "config.json"
    if config_file.exists() and Config(**json.loads(config_file.read_text())).as_dict() != config.as_dict():
        raise ValueError("This output directory belongs to a different configuration. Choose a new --output directory.")
    config_file.write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    started = time.perf_counter()
    all_metrics, validation = {}, {}
    robustness = None
    cases = ["soft", "flat", "sine"] if case == "suite" else [case]
    for model_case in cases:
        data, background, images, metrics = run_case(config, model_case, output, args.workers,
                                                    with_dsm=not args.skip_dsm, reuse_data=args.reuse_data)
        all_metrics[model_case] = metrics
        if model_case == "flat":
            validation.update(physical_checks(config, data, background))
            if validation["reciprocal_pairs"] == 0:
                validation["dedicated_reciprocity"] = reciprocity_check(config, output, args.workers)
        if model_case == "sine":
            robustness = run_robustness(config, data, background, output)
        # A usable report is saved even if a later independent experiment fails.
        write_report(config, all_metrics, output, robustness, validation)
    if args.refine:
        validation["mesh_refinement"] = refinement_check(config, output, args.workers)
    import numpy
    import scipy
    import matplotlib
    import numba
    manifest = dict(elapsed_seconds=time.perf_counter() - started,
                    python=platform.python_version(), platform=platform.platform(),
                    numpy=numpy.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                    numba=numba.__version__, cases=cases, workers=args.workers,
                    dsm_computed=not args.skip_dsm, mesh_refinement_computed=args.refine)
    manifest["config"] = config.as_dict()
    manifest["requested_case"] = args.case
    manifest["reuse_data"] = args.reuse_data
    for filename in ("metadata.json", "run_manifest.json"):
        (output / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_report(config, all_metrics, output, robustness, validation)
    print(f"Complete: {output.resolve()} ({manifest['elapsed_seconds']:.1f} s)", flush=True)
