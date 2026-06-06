#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Obstacle imaging experiments — direct / hybrid / joint-GN / limited-aperture."""
from __future__ import annotations

import argparse, sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from experiments.direct_imaging import main as run_direct
from experiments.hybrid_imaging import main as run_hybrid
from experiments.joint_gn import main as run_joint_gn
from experiments.limited_aperture_imaging import main as run_limited_aperture
from experiments.prior_sensitivity import main as run_prior_sensitivity
from experiments.apple_imaging import main as run_apple


def main():
    p = argparse.ArgumentParser(description="Obstacle imaging experiments.")
    p.add_argument("--mode", choices=["direct","hybrid","joint_gn","limited","prior","apple","all"],
                   default="all")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    modes = {
        "direct": run_direct, "hybrid": run_hybrid, "joint_gn": run_joint_gn,
        "limited": run_limited_aperture, "prior": run_prior_sensitivity, "apple": run_apple,
    }
    for name, fn in modes.items():
        if args.mode in (name, "all"):
            print(f"\n== {name} ==")
            fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
