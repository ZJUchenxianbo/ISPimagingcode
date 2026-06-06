#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1: polarimetric rank and conditioning."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ExperimentConfig, build_polarimetric_matrix, make_table, print_table


def run_experiment(config: ExperimentConfig) -> Any:
    """Reproduce the Table 1-style conditioning diagnostic."""
    rhos = [0.05, 0.25, 0.50, 0.75, 0.90, 0.98]
    if config.quick:
        rhos = [0.50, 0.90, 0.98]
    rows = []
    for rho in rhos:
        p = np.array([rho, 0.0, 0.0], dtype=float)
        row: dict[str, float | int] = {"rho": rho}
        for kind in ["full", "reciprocal"]:
            M = build_polarimetric_matrix(p, kind=kind, J=6)
            s = np.linalg.svd(M, compute_uv=False)
            tol = max(M.shape) * np.finfo(float).eps * float(s[0])
            row[f"sigma_min_{kind}"] = float(s[-1])
            row[f"cond_{kind}"] = float(s[0] / s[-1])
            row[f"rank_{kind}"] = int(np.sum(s > tol))
        rows.append(row)
    df = make_table(rows)
    df.to_csv(config.out_dir / "table1_polarimetric_conditioning.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.semilogy(df["rho"], df["cond_full"], marker="o", label="full")
    ax.semilogy(df["rho"], df["cond_reciprocal"], marker="s", label="reciprocal")
    ax.set_xlabel(r"$|p|$")
    ax.set_ylabel("condition number")
    ax.set_title("Polarimetric conditioning")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(config.out_dir / "polarimetric_conditioning.png", dpi=180)
    plt.close(fig)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the polarimetric conditioning diagnostic.")
    p.add_argument("--out-dir", type=str, default="outputs_section8_diagnostics")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_experiment(ExperimentConfig(out_dir=out_dir, seed=int(args.seed), quick=bool(args.quick)))
    print_table("Table 1: polarimetric rank and conditioning", df)


if __name__ == "__main__":
    main()
