#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 5: compare forward data sources at fixed wavenumber.

The three rows use Analytic Born, Discrete VIE-Born, and Full VIE far-field
data.  All rows share ``k=12``, the same finite-direction mock geometries, the
same standard complex noise sample at relative level 0.2, and the same four
inverse reconstruction methods.
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
        BasisComparisonRow(k=12.0, label="Analytic Born", data_source="analytic_born"),
        BasisComparisonRow(k=12.0, label="Discrete VIE-Born", data_source="vie_born"),
        BasisComparisonRow(k=12.0, label="Full VIE", data_source="full_vie"),
    ]
    return run_basis_comparison(
        config,
        experiment_number=5,
        row_specs=rows,
        output_stem="exp5_forward_comparison",
        figure_title="k = 12, noise = 0.2",
        noise_level=0.2,
        shared_noise_for_equal_k=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 5: forward-data comparison",
    )
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
