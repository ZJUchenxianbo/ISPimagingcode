#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward solvers for Maxwell far-field data.

Born approximation: analytical and voxel-discretised forms.
Full Maxwell VIE: volume integral equation with GMRES.
"""
from forward.born import (
    maxwell_born_far_field_fourier_formula,
)

from forward.vie import (
    VIESolveInfo,
    assemble_vie_matrix,
    ball_voxel_grid,
    dyadic_green_tensor,
    incident_plane_wave,
    maxwell_born_far_field,
    maxwell_far_field,
    solve_total_field_vie,
    tensor_ball_contrast,
    tensor_blocks_contrast,
    vie_to_fourier_convention,
)
