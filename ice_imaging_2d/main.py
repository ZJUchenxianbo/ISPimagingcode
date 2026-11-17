"""Single entry point for reproducible two-dimensional ice imaging experiments."""

import os
# Small matrix products should not start a thread pool inside each forward worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

# Support both direct-script and package execution without ambiguous common imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ice_imaging_2d.ice2d.config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["status", "backpropagate", "data", "indicator", "reconstruct"], default=None)
    parser.add_argument("--strategy", choices=["frozen", "full"], default="frozen")
    parser.add_argument("--model-budget", type=int, default=18)
    parser.add_argument("--mode", choices=["direct", "sensitivity", "legacy", "dual"], default="direct",
                        help="direct: proposal status/Cauchy backpropagation; sensitivity: explicit comparison; legacy: old acoustic comparison")
    parser.add_argument("--observed", help="Observed Cauchy data NPZ")
    parser.add_argument("--reference-data", help="Reference Cauchy data NPZ")
    parser.add_argument("--green", help="Validated causal coupled Gpp kernel NPZ")
    parser.add_argument("--experiment", choices=["bottom_sine", "bottom_bump", "top", "joint", "all"], default="joint")
    parser.add_argument("--quick", action="store_true", help="Use the small low-frequency dual_smoke configuration")
    parser.add_argument("--window", choices=["bottom", "internal", "full"], help="Fixed reference time window")
    parser.add_argument("--compare-windows", action="store_true", help="Run independent inversions with all three windows")
    parser.add_argument("--no-baselines", action="store_true", help="Skip legacy comparisons in dual experiments")
    parser.add_argument("--config", help="JSON configuration; default dual_short (dual) or compact (legacy)")
    parser.add_argument("--case", choices=["suite", "soft", "flat", "sine", "bump", "multiscale", "top_shift", "thick"], default=None)
    parser.add_argument("--output", "--out-dir", default="ice_outputs",
                        help="Output directory relative to the working directory")
    parser.add_argument("--workers", type=int, default=2, help="Independent forward shots in separate processes")
    parser.add_argument("--skip-dsm", action="store_true", help="Compute only the independent travel-time images")
    parser.add_argument("--reuse-data", action="store_true", help="Re-image existing matching data_*.npz files")
    parser.add_argument("--refine", action="store_true", help="Also run the 1 cm / 0.5 cm flat-model single-shot check")
    parser.add_argument("--estimate", action="store_true", help="Print grid and array sizes without running or writing files")
    args = parser.parse_args()
    if args.mode == "dual":
        parser.error("Old dual mode was numerical sensitivity, not the proposal method. Use --mode sensitivity explicitly for that comparison.")
    args.stage = args.stage or ("status" if args.mode == "direct" else "data")
    if args.mode == "direct":
        if args.quick or args.config or args.window or args.compare_windows or args.case or args.no_baselines or args.skip_dsm or args.reuse_data or args.refine:
            parser.error("Direct mode accepts supplied Cauchy data and kernels; forward/sensitivity flags are not applicable")
        from ice_imaging_2d.ice2d.direct_workflow import run_direct
        try:
            run_direct(args)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        return
    if args.stage in ("status", "backpropagate") or args.green or args.observed or args.reference_data:
        parser.error("status/backpropagate and Cauchy inputs require --mode direct")
    if args.model_budget < 14:
        parser.error("--model-budget must include at least 14 initial models")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode == "sensitivity":
        if args.case is not None:
            parser.error("--case belongs to --mode legacy; use --experiment for dual mode")
        if args.refine:
            parser.error("Use experiment.truth_grid_ratio in a dual config for distinct observation grids")
        if args.quick and args.config:
            parser.error("Choose either --quick or --config")
        if args.window and args.compare_windows:
            parser.error("Choose either --window or --compare-windows")
        from ice_imaging_2d.ice2d.dual_config import read_dual_config, estimate_cost
        config_path = args.config or Path(__file__).parent / "configs" / ("dual_smoke.json" if args.quick else "dual_short.json")
        physics, inverse, experiment = read_dual_config(config_path)
        if args.window:
            inverse = replace(inverse, window=args.window)
        if args.no_baselines:
            experiment = replace(experiment, baselines=False)
        names = ["bottom_sine", "bottom_bump", "top", "joint"] if args.experiment == "all" else [args.experiment]
        windows = ["bottom", "internal", "full"] if args.compare_windows else [inverse.window]
        if args.estimate:
            initial_models = 2 if args.stage == "data" else 2 + 2 * (len(inverse.control_x) + 1)
            print(f"Stage {args.stage}: initial cold-cache models={initial_models}; "
                  f"shots={initial_models * physics.source_count}; strategy={args.strategy}; "
                  f"frozen model budget={args.model_budget}")
            estimates = {name: estimate_cost(physics, inverse, experiment, name) for name in names}
            for value in estimates.values():
                if args.stage != "reconstruct" or args.strategy == "frozen":
                    value["forward_shots_upper_estimate"] = (args.model_budget if args.stage == "reconstruct" else initial_models) * physics.source_count
                    value["note"] = "Stage budget; excludes validation, optional legacy baselines and JIT. Cache hits may reduce actual solves."
            print(json.dumps(estimates, indent=2))
            print(f"Window runs per experiment: {len(windows)}; workers: {args.workers}")
            return
        estimate = estimate_cost(physics, inverse, experiment, names[0])
        print(f"Sensitivity comparison: grid {estimate['grid']}, {estimate['steps']} time steps/shot, "
              f"{physics.source_count} shots/model, workers={args.workers}", flush=True)
        if not args.quick:
            print("Research configuration: repeated CPU forward solves can be expensive. "
                  "Use --quick for a local smoke run; --estimate only reports sizes.", flush=True)
        from ice_imaging_2d.experiments.dual_interface import run_dual_experiment
        for name in names:
            for window in windows:
                directory = Path(args.output) / "runs" / name / args.stage / args.strategy / (args.window or "full")
                if args.compare_windows:
                    directory /= window
                run_dual_experiment(physics, replace(inverse, window=window), experiment, name,
                                    directory, args.workers, args.reuse_data, args.skip_dsm,
                                    args.stage, args.strategy, args.model_budget, Path(args.output))
        return
    if args.quick or args.window or args.compare_windows or args.no_baselines:
        parser.error("--quick/--window/--compare-windows/--no-baselines require --mode sensitivity")
    args.case = args.case or "suite"
    config = read_config(args.config)
    case = args.case
    if case == "top_shift":
        config = replace(config, ice_top=0.3).validate()
        case = "sine"
    elif case == "thick":
        config = replace(config, mean_bottom=-4.0, array_depth=5.5, seabed_depth=8.0,
                         image_z_min=-4.4, image_z_max=-3.6, record_seconds=0.007).validate()
        case = "sine"
    if args.estimate:
        dt = config.time_step
        stride = max(1, round(config.output_interval / dt))
        nt = (math.ceil(config.record_seconds / dt) - 1) // stride + 1
        nr = round(2 * config.receiver_halfwidth / config.receiver_spacing) + 1
        samples = config.source_count * nr * nt
        estimate = dict(grid_nx=round(2 * config.horizontal_halfwidth / config.grid_spacing) + 4,
                        grid_nz=round((config.computational_top + config.seabed_depth) / config.grid_spacing) + 4,
                        time_step_seconds=dt, recorded_time_step_seconds=stride * dt,
                        shots=config.source_count, receivers=nr, recorded_samples=nt,
                        one_case_pressure_and_gradient_gib=8 * samples / 2 ** 30,
                        approximate_postprocessing_arrays_gib=80 * samples / 2 ** 30,
                        note="Postprocessing estimate excludes FFT temporaries, DSM kernels and forward worker arrays.")
        print(json.dumps(estimate, indent=2))
        return
    # Only actual experiments load NumPy/SciPy, plotting and Numba kernels.
    from ice_imaging_2d.experiments.run import run_experiments

    run_experiments(config, case, args)


if __name__ == "__main__":
    main()
