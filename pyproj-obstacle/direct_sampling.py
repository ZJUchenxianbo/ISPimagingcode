#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared direct-sampling workflow for sound-soft obstacle experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.scattering import Array, CArray, add_relative_noise
from common.forward import solve_forward_farfield
from common.sampling import direct_sampling_indicator


@dataclass(frozen=True)
class DirectSamplingResult:
    """Clean/noisy far fields and images for one obstacle target case."""
    farfield_clean: CArray
    farfield_noisy_list: list[CArray]
    image_clean: Array
    image_noisy_list: list[Array]


def compute_direct_sampling_result(
    params: Array,
    n_obstacles: int,
    n_boundary: int,
    k: float,
    incident_angles: Array,
    obs_angles: Array,
    x_grid: Array,
    y_grid: Array,
    aperture_length: float,
    noise_levels: Array,
    seed: int,
    *,
    indicator_power: float = 1.0,
    block_size: int = 32768,
    noise_seed_stride: int = 1000,
) -> DirectSamplingResult:
    """Compute clean and noisy direct-sampling images for one target.

    The mathematical model is unchanged: first solve the sound-soft obstacle
    forward problem, then evaluate the multi-incident direct/orthogonality
    sampling indicator with the supplied observation aperture.
    """
    farfield_clean = solve_forward_farfield(
        params,
        k,
        n_boundary,
        incident_angles,
        obs_angles,
        n_obstacles=n_obstacles,
    )
    image_clean = direct_sampling_indicator(
        farfield_clean,
        k,
        obs_angles,
        incident_angles,
        x_grid,
        y_grid,
        aperture_length,
        power=indicator_power,
        block_size=block_size,
    )

    farfield_noisy_list: list[CArray] = []
    image_noisy_list: list[Array] = []
    for idx, noise_level in enumerate(noise_levels):
        rng = np.random.default_rng(int(seed) + int(noise_seed_stride) * idx)
        farfield_noisy = add_relative_noise(farfield_clean, float(noise_level), rng)
        image_noisy = direct_sampling_indicator(
            farfield_noisy,
            k,
            obs_angles,
            incident_angles,
            x_grid,
            y_grid,
            aperture_length,
            power=indicator_power,
            block_size=block_size,
        )
        farfield_noisy_list.append(farfield_noisy)
        image_noisy_list.append(image_noisy)

    return DirectSamplingResult(
        farfield_clean=farfield_clean,
        farfield_noisy_list=farfield_noisy_list,
        image_clean=image_clean,
        image_noisy_list=image_noisy_list,
    )
