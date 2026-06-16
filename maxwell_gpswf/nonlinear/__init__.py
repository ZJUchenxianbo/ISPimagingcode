#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonlinear Maxwell inversion helpers."""

from nonlinear.bim_gpswf import (
    BIMLinearization,
    ScalarVIEData,
    build_scalar_contrast,
    compute_bim_gpswf_linearization,
    compute_scalar_vie_data,
    evaluate_blocks_on_nodes,
    solve_tikhonov_update,
)

