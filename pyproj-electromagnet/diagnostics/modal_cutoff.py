#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 4: modal spectral-cutoff stability test."""
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
    collect_alpha_pairs_cached,
    complex_relative_noise,
    make_table,
    print_table,
    vector_norm,
)


def run_experiment(config: ExperimentConfig) -> Any:
    """Reproduce the Table 4-style modal spectral-cutoff diagnostic."""
    rng = np.random.default_rng(config.seed + 20)
    C = 20.0
    K = 80
    ell_max = 28
    n_modes_per_ell = 16
    quad_order = 220
    r_eval_count = 160
    num_trials = 200
    if config.quick:
        K = 40
        ell_max = 8
        n_modes_per_ell = 8
        quad_order = 90
        r_eval_count = 70
        num_trials = 40

    alpha_df = collect_alpha_pairs_cached(
        C,
        K,
        ell_max,
        n_modes_per_ell,
        quad_order=quad_order,
        r_eval_count=r_eval_count,
        cache_dir=config.out_dir / "alpha_cache",
    )
    alpha_df.to_csv(config.out_dir / "alpha_estimates.csv", index=False)
    alpha = alpha_df["alpha_real"].to_numpy(dtype=float) + 1j * alpha_df["alpha_imag"].to_numpy(dtype=float)
    alpha_abs = np.abs(alpha)
    alpha_max = max(float(np.max(alpha_abs)), 1e-300)

    # Low-rank diagnostic signal: true modal coefficients decay with the
    # Fourier eigenvalue magnitude, so the table emphasizes spectral cutoff
    # stability rather than arbitrary high-frequency truncation error.
    modal_envelope = (alpha_abs / alpha_max) ** 2

    epsilon_ratios = [1e-1, 1e-2, 1e-3]
    noise_levels = [1e-4, 1e-3, 1e-2]
    rows = []
    for epsilon_ratio in epsilon_ratios:
        epsilon = float(epsilon_ratio) * alpha_max
        retained = alpha_abs > epsilon
        errors = {delta: [] for delta in noise_levels}
        for _ in range(num_trials):
            Q = modal_envelope * (rng.normal(size=alpha.shape) + 1j * rng.normal(size=alpha.shape))
            W = alpha * Q
            for delta in noise_levels:
                eta = complex_relative_noise(W, delta, rng)
                W_delta = W + eta
                Q_rec = np.zeros_like(Q)
                Q_rec[retained] = W_delta[retained] / alpha[retained]
                errors[delta].append(vector_norm(Q_rec - Q) / vector_norm(Q))
        row: dict[str, float | int] = {
            "epsilon_ratio": epsilon_ratio,
            "epsilon": epsilon,
            "retained_pairs": int(np.sum(retained)),
            "alpha_max": alpha_max,
            "theory_delta_over_epsilon_ratio_for_1e-3": 1e-3 / epsilon_ratio,
        }
        for delta in noise_levels:
            row[f"median_error_{delta:g}"] = float(np.median(errors[delta]))
        rows.append(row)

    df = make_table(rows)
    df.to_csv(config.out_dir / "table4_modal_cutoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for delta in noise_levels:
        ax.loglog(df["epsilon_ratio"], df[f"median_error_{delta:g}"], marker="o", label=fr"$\delta={delta:g}$")
    ax.invert_xaxis()
    ax.set_xlabel(r"cutoff ratio $\epsilon / \max_{\ell,n}|\alpha_{\ell n}|$")
    ax.set_ylabel("median relative modal error")
    ax.set_title("Modal spectral cutoff")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(config.out_dir / "modal_cutoff_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for ell in [0, min(4, ell_max), min(8, ell_max)]:
        sub = alpha_df[alpha_df["ell"] == ell]
        ax.semilogy(sub["n"], sub["alpha_abs"], marker="o", label=fr"$\ell={ell}$")
    ax.set_xlabel("radial mode n")
    ax.set_ylabel(r"$|\alpha_{\ell,n}|$")
    ax.set_title("Estimated alpha decay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(config.out_dir / "alpha_decay.png", dpi=180)
    plt.close(fig)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the modal spectral-cutoff diagnostic.")
    p.add_argument("--out-dir", type=str, default="outputs_section8_diagnostics")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_experiment(ExperimentConfig(out_dir=out_dir, seed=int(args.seed), quick=bool(args.quick)))
    print_table("Table 4: modal spectral cutoff", df)


if __name__ == "__main__":
    main()
