#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 4: compare four reconstruction methods across wavenumbers.

Rows use ``k = 8, 12, 15``.  Every row is generated from finite-direction
Full VIE far-field data with relative far-field noise 0.2.  Columns are GPSWF,
cube Fourier, ball Bessel, and DSM.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.basis_comparison import (
    BasisComparisonRow,
    run_basis_comparison,
)
from common.config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> Any:
    rows = [
        BasisComparisonRow(k=float(k), label=f"k = {k:g}", data_source="full_vie")
        for k in (8, 12, 15)
    ]
    return run_basis_comparison(
        config,
        experiment_number=4,
        row_specs=rows,
        output_stem="exp4_basis",
        figure_title="noise = 0.2",
        noise_level=0.2,
        shared_noise_for_equal_k=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4: basis comparison")
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(ExperimentConfig(
        out_dir=out_dir,
        seed=args.seed,
        quick=args.quick,
    ))


if __name__ == "__main__":
    main()
