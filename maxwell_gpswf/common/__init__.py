#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for Maxwell-Born inversion diagnostics.

This package re-exports all symbols that were previously defined in the
monolithic ``common.py``, so existing import statements continue to work.
"""
from common.config import ExperimentConfig

from common.phantom import (
    Block,
    Mode,
    TensorBlock,
    TensorKind,
    ball_phantom,
    block_fourier_profile,
    cube_phantom,
    dispersed_blocks_phantom,
    normalized_ball_fourier_profile,
    reference_tensor,
    sphere_fourier_profile,
    sphere_truth_2d,
    tensor_basis,
    tensor_block_fourier_coefficients,
    tensor_truth_image_2d,
    tensor_coefficients_from_matrix,
    three_block_phantom,
    three_tensor_block_phantom,
    truth_image_2d,
    two_spheres_cube_phantom,
    _shape_truth_and_fourier,
)

from common.quadrature import (
    LEBEDEV_ORDERS_AND_COUNTS,
    SphereRule,
    admissible_farfield_pairs_from_nodes,
    ball_quadrature_nodes,
    build_geometries_from_p,
    equal_area_sphere_directions,
    farfield_fourier_nodes,
    fibonacci_sphere_directions,
    generate_data_nodes,
    interior_ball_nodes,
    lebedev_sphere_quadrature,
    match_mock_quadrature_nodes,
    orthonormal_basis_perp,
    paired_farfield_fourier_nodes,
    sphere_quadrature,
)

from common.polarimetric import (
    build_polarimetric_matrix,
    polarimetric_farfield_data,
    recover_polarimetric_coefficients,
    recover_polarimetric_coefficients_from_data,
)

from common.gpswf import (
    ball_gpswf_tridiagonal,
    collect_alpha_pairs,
    collect_alpha_pairs_cached,
    compute_alpha_radial,
    eval_radial_R,
    jacobi_orthonormal_scale,
    modal_matrix,
    quadrature_modal_coefficients,
    solve_ball_gpswf,
    spherical_coordinates,
    tridiagonal_residual,
)

from common.fourier import (
    block_fourier_transform_xi,
    cube_fourier_data_matrix,
    cube_fourier_coefficients_from_blocks,
    cube_fourier_frequencies,
    cube_fourier_indices,
    evaluate_cube_fourier_series,
    reconstruct_fourier_cube_from_data,
    reconstruct_blocks_fourier_cube,
)

from common.spherical_bessel import (
    BesselMode,
    ball_bessel_coefficients_from_blocks,
    ball_bessel_data_matrix,
    ball_bessel_matrix,
    ball_bessel_modes,
    block_values_at_points,
    reconstruct_ball_bessel_from_data,
    reconstruct_blocks_ball_bessel,
    spherical_bessel_roots,
)

from common.direct_sampling import (
    direct_sampling_component_indicator,
    direct_sampling_farfield_indicator,
    direct_sampling_tensor_indicator,
)

from common.diagnostics import (
    collect_reconstruction_diagnostics,
    coefficient_diagnostics,
    data_diagnostics,
    image_diagnostics,
    mode_diagnostics,
    node_diagnostics,
    plot_diagnostic_curves,
    projection_diagnostics,
    save_diagnostics_npz,
    write_diagnostics_csv,
)

from common.utils import (
    Array,
    CArray,
    SimpleColumn,
    SimpleTable,
    complex_relative_noise,
    make_table,
    print_table,
    vector_norm,
    weighted_lstsq,
)
