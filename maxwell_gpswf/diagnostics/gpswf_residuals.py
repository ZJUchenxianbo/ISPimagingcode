#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3: ball GPSWF eigensystem residual diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    ExperimentConfig,
    ball_gpswf_tridiagonal,
    collect_alpha_pairs_cached,
    make_table,
    print_table,
    solve_ball_gpswf,
    tridiagonal_residual,
)


def run_experiment(config: ExperimentConfig) -> Any:
    """Reproduce the Table 3-style GPSWF residual diagnostic."""
    settings = [
        {"C": 10.0, "K": 60, "ell_max": 18},
        {"C": 20.0, "K": 80, "ell_max": 28},
        {"C": 30.0, "K": 100, "ell_max": 38},
    ]
    if config.quick:
        settings = [
            {"C": 10.0, "K": 30, "ell_max": 8},
            {"C": 20.0, "K": 40, "ell_max": 12},
        ]
    n_alpha_modes = 20
    quad_order = 220
    r_eval_count = 160
    if config.quick:
        n_alpha_modes = 10
        quad_order = 90
        r_eval_count = 70
    rows = []
    for setting in settings:
        C = float(setting["C"])
        K = int(setting["K"])
        ell_max = int(setting["ell_max"])
        max_residual = 0.0
        for ell in range(ell_max + 1):
            chi, beta = solve_ball_gpswf(C, ell, K, n_modes=min(20, K + 1))
            diag, offdiag = ball_gpswf_tridiagonal(C, ell, K)
            for n in range(beta.shape[1]):
                max_residual = max(max_residual, tridiagonal_residual(diag, offdiag, chi[n], beta[:, n]))
        alpha_df = collect_alpha_pairs_cached(
            C,
            K,
            ell_max,
            n_alpha_modes,
            quad_order=quad_order,
            r_eval_count=r_eval_count,
            cache_dir=config.out_dir / "alpha_cache",
        )
        alpha_abs = alpha_df["alpha_abs"].to_numpy(dtype=float)
        alpha_max = max(float(alpha_abs.max()), 1e-300)
        rows.append(
            {
                "C": C,
                "K": K,
                "ell_max": ell_max,
                "max_residual": max_residual,
                "max_alpha_abs": alpha_max,
                "pairs_above_1e-2_max": int((alpha_abs > 1e-2 * alpha_max).sum()),
                "pairs_above_1e-3_max": int((alpha_abs > 1e-3 * alpha_max).sum()),
            }
        )
    df = make_table(rows)
    df.to_csv(config.out_dir / "table3_gpswf_residuals.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the GPSWF eigensystem residual diagnostic.")
    p.add_argument("--out-dir", type=str, default="outputs_section8_diagnostics")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_experiment(ExperimentConfig(out_dir=out_dir, seed=int(args.seed), quick=bool(args.quick)))
    print_table("Table 3: GPSWF eigensystem residuals", df)


if __name__ == "__main__":
    main()
