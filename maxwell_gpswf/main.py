#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Maxwell GPSWF imaging experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ExperimentConfig
from experiments.exp1_dimension import run_experiment as run_exp1
from experiments.exp2_noise import run_experiment as run_exp2
from experiments.exp3_frequency import run_experiment as run_exp3
from experiments.exp4_basis import run_experiment as run_exp4
from experiments.exp5_forward_comparison import run_experiment as run_exp5


def parse_args():
    p = argparse.ArgumentParser(description="Run Maxwell far-field imaging experiments.")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--mode",
        choices=["exp1", "exp2", "exp3", "exp4", "exp5", "all"],
        default="all",
        help=(
            "'all' runs the Full VIE experiments exp1-exp4; "
            "run exp5 explicitly"
        ),
    )
    return p.parse_args()


def _subdir(base: Path, name: str) -> Path:
    d = base / name; d.mkdir(parents=True, exist_ok=True); return d


def main():
    args = parse_args()
    base = Path(args.out_dir); base.mkdir(parents=True, exist_ok=True)

    def cfg(sub: str) -> ExperimentConfig:
        return ExperimentConfig(out_dir=_subdir(base, sub), seed=args.seed,
                                quick=args.quick)

    if args.mode in {"exp1", "all"}:
        print("\n== Experiment 1: Dimension effects (single cube) ==")
        run_exp1(cfg("exp1"))

    if args.mode in {"exp2", "all"}:
        print("\n== Experiment 2: Noise effects (three blocks) ==")
        run_exp2(cfg("exp2"))

    if args.mode in {"exp3", "all"}:
        print("\n== Experiment 3: Frequency effects (three blocks) ==")
        run_exp3(cfg("exp3"))

    if args.mode in {"exp4", "all"}:
        print("\n== Experiment 4: Basis comparison ==")
        run_exp4(cfg("exp4"))

    if args.mode == "exp5":
        print("\n== Experiment 5: Forward-data comparison at k=12 ==")
        run_exp5(cfg("exp5"))

    print(f"\nDone. Output: {base}")


if __name__ == "__main__":
    main()
