#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared library for 2D acoustic and electromagnetic scattering experiments.

Layer 0 (no project dependencies):
  - scattering: constants, type aliases, noise helpers, direction vectors

Layer 1 (import Layer 0):
  - forward: BEM / Lippmann-Schwinger / point-scatterer solvers
  - sampling: direct-sampling indicators, aperture utilities
  - targets: synthetic target case definitions
  - unet: U-Net model for learned imaging enhancement
"""
from common.scattering import (
    PI2, Array, CArray,
    direction_vectors, parse_float_list,
    add_relative_noise, empirical_snr, safe_slug,
)
from common.forward import (
    ObstacleMethod, MediumMethod, PointMethod,
    solve_obstacle_farfield, solve_forward_farfield,
    solve_point_scatterer_farfield, solve_medium_farfield,
    plane_wave, plane_wave_matrix,
    build_single_layer_matrix, single_layer_farfield_operator,
    build_double_layer_matrix, double_layer_farfield_operator,
)
from common.sampling import (
    aperture_measure, aperture_angles, observation_weights,
    normalize_indicator, image_extent, plot_indicator_image,
    direct_sampling_indicator, direct_sampling_indicators,
)
from common.targets import (
    ObstacleTargetCase, MediumTargetCase, PointScattererCase,
    BoundaryGeometry, params_to_geometry,
    impenetrable_obstacle_cases, limited_aperture_obstacle_cases,
    medium_cases, point_scatterer_cases,
    star_boundary, dense_boundary_points,
    parse_case_names, plot_obstacle_boundaries, deduplicate_legend,
)
from common.unet import (
    PointScattererUNetConfig, UNet2D,
    point_support_heatmap,
    build_point_scatterer_unet_dataset,
    train_unet_from_arrays, train_point_scatterer_unet,
    predict_unet_images, save_unet_checkpoint,
)
