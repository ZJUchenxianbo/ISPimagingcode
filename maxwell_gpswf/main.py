#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Maxwell GPSWF imaging experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ExperimentConfig
from experiments.figure1_noise_dimension import run_experiment as run_fig1
from experiments.figure2_frequency_contrast import run_experiment as run_fig2
from experiments.figure3_sources_shapes import run_experiment as run_fig3
from experiments.figure4_scale_scaling import run_experiment as run_fig4
from experiments.figure5_basis_comparison import run_experiment as run_fig5
from experiments.figure6_tensor_blocks import run_experiment as run_fig6
from experiments.figure7_bim_gpswf_frequency import run_experiment as run_fig7


def parse_args():
    p = argparse.ArgumentParser(description="Run Maxwell-Born imaging experiments.")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-mode", choices=["mock", "ideal"], default="mock",
                   help="mock: 06005-style nearest measured node; ideal: admissible direction pairs")
    p.add_argument("--mode", choices=["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "all"], default="all")
    return p.parse_args()


def _subdir(base: Path, name: str) -> Path:
    d = base / name; d.mkdir(parents=True, exist_ok=True); return d


def main():
    args = parse_args()
    base = Path(args.out_dir); base.mkdir(parents=True, exist_ok=True)

    def cfg(sub: str) -> ExperimentConfig:
        return ExperimentConfig(out_dir=_subdir(base, sub), seed=args.seed,
                                quick=args.quick, data_mode=args.data_mode)

    if args.mode in {"fig1", "all"}:
        print(f"\n== Figure 1: Noise & dimension [{args.data_mode}] ==")
        run_fig1(cfg("fig1"))

    if args.mode in {"fig2", "all"}:
        print(f"\n== Figure 2: Frequency & contrast [{args.data_mode}] ==")
        run_fig2(cfg("fig2"))

    if args.mode in {"fig3", "all"}:
        print(f"\n== Figure 3: Sources & shapes [{args.data_mode}] ==")
        run_fig3(cfg("fig3"))

    if args.mode in {"fig4", "all"}:
        print(f"\n== Figure 4: Scale scaling [{args.data_mode}] ==")
        run_fig4(cfg("fig4"))

    if args.mode in {"fig5", "all"}:
        print(f"\n== Figure 5: Basis comparison [{args.data_mode}] ==")
        run_fig5(cfg("fig5"))

    if args.mode in {"fig6", "all"}:
        print(f"\n== Figure 6: Tensor block reconstruction [{args.data_mode}] ==")
        run_fig6(cfg("fig6"))

    if args.mode in {"fig7", "all"}:
        print(f"\n== Figure 7: BIM-GPSWF frequency experiment [{args.data_mode}] ==")
        run_fig7(cfg("fig7"))

    print(f"\nDone. Output: {base}")


if __name__ == "__main__":
    main()
