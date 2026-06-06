#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2: noise amplification in the polarimetric solve."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig,
    build_polarimetric_matrix,
    complex_relative_noise,
    make_table,
    print_table,
    vector_norm,
)


def run_experiment(config: ExperimentConfig) -> Any:
    """Reproduce the Table 2-style polarimetric noise diagnostic."""
    rng = np.random.default_rng(config.seed + 10)
    rhos = [0.50, 0.90, 0.98]
    noise_levels = [1e-4, 1e-3, 1e-2, 5e-2]
    num_trials = 80 if config.quick else 500
    rows = []
    for rho in rhos:
        M = build_polarimetric_matrix(np.array([rho, 0.0, 0.0]), kind="full", J=6)
        s = np.linalg.svd(M, compute_uv=False)
        M_pinv = np.linalg.pinv(M)
        cond = float(s[0] / s[-1])
        trial_errors = {delta: [] for delta in noise_levels}
        for _ in range(num_trials):
            c = rng.normal(size=M.shape[1]) + 1j * rng.normal(size=M.shape[1])
            g = M @ c
            for delta in noise_levels:
                eta = complex_relative_noise(g, delta, rng)
                c_rec = M_pinv @ (g + eta)
                trial_errors[delta].append(vector_norm(c_rec - c) / vector_norm(c))
        row: dict[str, float] = {"rho": rho, "cond": cond}
        for delta in noise_levels:
            row[f"median_error_{delta:g}"] = float(np.median(trial_errors[delta]))
        rows.append(row)
    df = make_table(rows)
    df.to_csv(config.out_dir / "table2_noise_amplification.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for _, row in df.iterrows():
        vals = [row[f"median_error_{delta:g}"] for delta in noise_levels]
        ax.loglog(noise_levels, vals, marker="o", label=fr"$\rho={row['rho']:.2f}$")
    ax.set_xlabel("relative noise level")
    ax.set_ylabel("median relative coefficient error")
    ax.set_title("Polarimetric noise amplification")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(config.out_dir / "noise_amplification.png", dpi=180)
    plt.close(fig)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the polarimetric noise amplification diagnostic.")
    p.add_argument("--out-dir", type=str, default="outputs_section8_diagnostics")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_experiment(ExperimentConfig(out_dir=out_dir, seed=int(args.seed), quick=bool(args.quick)))
    print_table("Table 2: noise amplification", df)


if __name__ == "__main__":
    main()
