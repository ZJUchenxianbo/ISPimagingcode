#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment configuration shared across all scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime settings shared by the Maxwell GPSWF experiments."""

    out_dir: Path
    seed: int = 12345
    quick: bool = False
