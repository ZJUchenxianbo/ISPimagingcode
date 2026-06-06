#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Maxwell imaging experiments and diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ExperimentConfig, print_table
from experiments.figure1_noise_dimension import run_experiment as run_fig1
from experiments.figure2_frequency_contrast import run_experiment as run_fig2
from experiments.figure3_sources_shapes import run_experiment as run_fig3
from diagnostics.gpswf_residuals import run_experiment as run_gpswf_residuals
from diagnostics.modal_cutoff import run_experiment as run_modal_cutoff
from diagnostics.noise_amplification import run_experiment as run_noise_amplification
from diagnostics.polarimetric_conditioning import run_experiment as run_polarimetric_conditioning


def parse_args():
    p = argparse.ArgumentParser(description="Run Maxwell-Born imaging experiments.")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mode", choices=["fig1", "fig2", "fig3", "diagnostics", "all"], default="all")
    return p.parse_args()


def _subdir(base: Path, name: str) -> Path:
    d = base / name; d.mkdir(parents=True, exist_ok=True); return d


def main():
    args = parse_args()
    base = Path(args.out_dir); base.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(out_dir=base, seed=args.seed, quick=args.quick)

    if args.mode in {"fig1", "all"}:
        print("\n== Figure 1: Noise & dimension ==")
        run_fig1(ExperimentConfig(out_dir=_subdir(base, "fig1"), seed=args.seed, quick=args.quick))

    if args.mode in {"fig2", "all"}:
        print("\n== Figure 2: Frequency & contrast ==")
        run_fig2(ExperimentConfig(out_dir=_subdir(base, "fig2"), seed=args.seed, quick=args.quick))

    if args.mode in {"fig3", "all"}:
        print("\n== Figure 3: Sources & shapes ==")
        run_fig3(ExperimentConfig(out_dir=_subdir(base, "fig3"), seed=args.seed, quick=args.quick))

    if args.mode in {"diagnostics", "all"}:
        dout = _subdir(base, "diagnostics")
        print("\n== Diagnostics ==")
        for name, fn in [("Table 1: polarimetric", run_polarimetric_conditioning),
                          ("Table 2: noise amplification", run_noise_amplification),
                          ("Table 3: GPSWF residuals", run_gpswf_residuals),
                          ("Table 4: modal cutoff", run_modal_cutoff)]:
            t = fn(ExperimentConfig(out_dir=dout, seed=args.seed, quick=args.quick))
            print_table(name, t)

    print(f"\nDone. Output: {base}")


if __name__ == "__main__":
    main()
