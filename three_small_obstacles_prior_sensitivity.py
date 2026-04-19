#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from three_small_obstacles_joint_gn_random_centers import (
    PI2,
    CaseMetrics,
    add_relative_noise,
    build_argparser as _unused_build_argparser,
    dense_boundary_points,
    empirical_snr,
    enforce_constraints,
    gauss_newton_reconstruct,
    generate_random_centers,
    music_indicator,
    obstacle_param_slice,
    pairwise_min_distance,
    params_to_geometry,
    parse_float_list,
    resolved_from_centers,
    save_case_plot,
    select_peaks_2d,
    solve_forward_farfield,
)


def centers_from_params(p: np.ndarray) -> np.ndarray:
    return np.array([[p[obstacle_param_slice(j).start], p[obstacle_param_slice(j).start + 1]] for j in range(3)], dtype=float)


def make_init_params(mode: str, img: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, spacing: float,
                     center_extent: float, init_radius: float, args: argparse.Namespace,
                     rng: np.random.Generator) -> np.ndarray:
    if mode == 'music':
        centers = select_peaks_2d(img, x_grid, y_grid, n_peaks=3, exclusion_radius=max(0.08, 0.45 * float(spacing)))
        blocks = [np.concatenate([centers[q], np.array([init_radius, 0.0, 0.0, 0.0, 0.0])]) for q in range(3)]
    elif mode == 'random_generic':
        centers = generate_random_centers(float(spacing), rng, center_extent, float(args.min_gap))
        rng.shuffle(centers, axis=0)
        blocks = [np.concatenate([centers[q], np.array([init_radius, 0.0, 0.0, 0.0, 0.0])]) for q in range(3)]
    elif mode == 'poor_clustered':
        # Deliberately poor: clustered centers near origin, wrong radius and nonzero wrong shape guesses.
        centers = np.array([
            [-0.05, -0.03],
            [ 0.03,  0.02],
            [ 0.07, -0.01],
        ], dtype=float)
        wrong_shapes = [
            np.array([0.060,  0.10, -0.10,  0.06, -0.05]),
            np.array([0.060, -0.12,  0.08, -0.04,  0.05]),
            np.array([0.060,  0.11,  0.09, -0.06,  0.04]),
        ]
        blocks = [np.concatenate([centers[q], wrong_shapes[q]]) for q in range(3)]
    else:
        raise ValueError(f'unknown init mode: {mode}')

    p_init = np.concatenate(blocks).astype(float)
    p_init = enforce_constraints(
        p_init,
        float(args.min_gap),
        (float(args.min_radius), float(args.max_radius)),
        (float(args.min_coeff), float(args.max_coeff)),
        center_extent,
    )
    return p_init


def save_panel(path: Path, p_true: np.ndarray, init_map: Dict[str, np.ndarray], rec_map: Dict[Tuple[str, float], np.ndarray],
               modes: List[str], noises: np.ndarray, d_true_min: float, d_rayleigh: float) -> None:
    nrows, ncols = len(noises), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.8*nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.shape != (nrows, ncols):
        if nrows == 1:
            axes = axes.reshape(1, ncols)
        elif ncols == 1:
            axes = axes.reshape(nrows, 1)
    for i, noise in enumerate(noises):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            for p, style, lw in [(p_true, 'k--', 1.3), (init_map[mode], 'b:', 1.0), (rec_map[(mode, float(noise))], 'r-', 1.3)]:
                for q in range(3):
                    pts = dense_boundary_points(p[obstacle_param_slice(q)])
                    ax.plot(pts[:,0], pts[:,1], style, lw=lw)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.15)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{mode}\nnoise={noise:.2f}')
    fig.suptitle(f'Prior sensitivity, true d_min={d_true_min:.3f} ({d_true_min/d_rayleigh:.2f} d_R)')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description='Prior sensitivity for three-obstacle joint GN reconstruction')
    p.add_argument('--out-dir', type=str, default='outputs_three_prior_sensitivity')
    p.add_argument('--k', type=float, default=8.0)
    p.add_argument('--radius', type=float, default=0.045)
    p.add_argument('--spacing', type=float, default=0.18)
    p.add_argument('--noise-levels', type=str, default='0.01,0.05,0.10')
    p.add_argument('--incident-angles', type=str, default='0,0.7853981634,1.5707963268,2.3561944902,3.1415926536,3.9269908170,4.7123889804,5.4977871438')
    p.add_argument('--n-per-obstacle', type=int, default=10)
    p.add_argument('--n-obs', type=int, default=10)
    p.add_argument('--grid-extent', type=float, default=0.45)
    p.add_argument('--grid-size', type=int, default=41)
    p.add_argument('--center-extent', type=float, default=0.22)
    p.add_argument('--init-radius', type=float, default=0.05)
    p.add_argument('--min-radius', type=float, default=0.03)
    p.add_argument('--max-radius', type=float, default=0.07)
    p.add_argument('--min-coeff', type=float, default=-0.18)
    p.add_argument('--max-coeff', type=float, default=0.18)
    p.add_argument('--min-gap', type=float, default=0.008)
    p.add_argument('--n-iter', type=int, default=2)
    p.add_argument('--lambda-reg', type=float, default=1.0e-2)
    p.add_argument('--damping', type=float, default=0.7)
    p.add_argument('--true1-a2c', type=float, default=0.12)
    p.add_argument('--true1-a2s', type=float, default=-0.08)
    p.add_argument('--true1-a3c', type=float, default=0.06)
    p.add_argument('--true1-a3s', type=float, default=0.03)
    p.add_argument('--true2-a2c', type=float, default=-0.10)
    p.add_argument('--true2-a2s', type=float, default=0.09)
    p.add_argument('--true2-a3c', type=float, default=-0.05)
    p.add_argument('--true2-a3s', type=float, default=0.04)
    p.add_argument('--true3-a2c', type=float, default=0.08)
    p.add_argument('--true3-a2s', type=float, default=0.10)
    p.add_argument('--true3-a3c', type=float, default=0.05)
    p.add_argument('--true3-a3s', type=float, default=-0.06)
    p.add_argument('--seed', type=int, default=24680)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = float(args.k)
    d_rayleigh = math.pi / k
    noises = parse_float_list(args.noise_levels)
    incident_angles = parse_float_list(args.incident_angles)
    obs_angles = np.linspace(0.0, PI2, int(args.n_obs), endpoint=False)
    x_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    y_grid = np.linspace(-float(args.grid_extent), float(args.grid_extent), int(args.grid_size))
    center_extent = float(args.center_extent)

    coeffs_true = [
        np.array([float(args.radius), float(args.true1_a2c), float(args.true1_a2s), float(args.true1_a3c), float(args.true1_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true2_a2c), float(args.true2_a2s), float(args.true2_a3c), float(args.true2_a3s)], dtype=float),
        np.array([float(args.radius), float(args.true3_a2c), float(args.true3_a2s), float(args.true3_a3c), float(args.true3_a3s)], dtype=float),
    ]

    rng_cent = np.random.default_rng(int(args.seed))
    centers_true = generate_random_centers(float(args.spacing), rng_cent, center_extent, float(args.min_gap))
    p_true = np.concatenate([np.concatenate([centers_true[q], coeffs_true[q]]) for q in range(3)]).astype(float)
    ff_clean = solve_forward_farfield(p_true, k, int(args.n_per_obstacle), incident_angles, obs_angles)
    np.savez_compressed(out_dir / 'farfield_clean.npz', farfield_clean=ff_clean, p_true=p_true, centers_true=centers_true, d_rayleigh=d_rayleigh)

    # one music image from clean data is enough for init on all noise levels? Use noisy-specific image for fairness.
    init_map: Dict[str, np.ndarray] = {}
    rec_map: Dict[Tuple[str, float], np.ndarray] = {}
    rows: List[dict] = []
    modes = ['music', 'random_generic', 'poor_clustered']

    for noise in noises:
        rng_noise = np.random.default_rng(int(args.seed) + int(round(1000*noise)))
        ff_noisy = add_relative_noise(ff_clean, float(noise), rng_noise)
        img = music_indicator(ff_noisy, k, obs_angles, x_grid, y_grid, rank_signal=3)
        for mode_idx, mode in enumerate(modes):
            rng_init = np.random.default_rng(int(args.seed) + 100*mode_idx + int(round(1000*noise)))
            p_init = make_init_params(mode, img, x_grid, y_grid, float(args.spacing), center_extent, float(args.init_radius), args, rng_init)
            p_rec, history = gauss_newton_reconstruct(
                ff_noisy, p_init, k=k, n_per_obstacle=int(args.n_per_obstacle),
                incident_angles=incident_angles, obs_angles=obs_angles, n_iter=int(args.n_iter),
                lambda_reg=float(args.lambda_reg), damping=float(args.damping),
                radius_bounds=(float(args.min_radius), float(args.max_radius)),
                coeff_bounds=(float(args.min_coeff), float(args.max_coeff)),
                min_gap=float(args.min_gap), center_extent=center_extent,
            )
            if mode not in init_map:
                init_map[mode] = p_init.copy()
            rec_map[(mode, float(noise))] = p_rec.copy()
            centers_init = centers_from_params(p_init)
            centers_rec = centers_from_params(p_rec)
            resolved, mean_err, max_err = resolved_from_centers(centers_true, centers_rec, pairwise_min_distance(centers_true))
            ff_rec = solve_forward_farfield(p_rec, k, int(args.n_per_obstacle), incident_angles, obs_angles)
            rel_res = float(np.linalg.norm(ff_rec - ff_noisy) / max(np.linalg.norm(ff_noisy), 1e-14))
            row = {
                'init_mode': mode,
                'noise': float(noise),
                'spacing_true_min': pairwise_min_distance(centers_true),
                'spacing_init_min': pairwise_min_distance(centers_init),
                'spacing_rec_min': pairwise_min_distance(centers_rec),
                'd_rayleigh': d_rayleigh,
                'spacing_over_dR': pairwise_min_distance(centers_true)/d_rayleigh,
                'srf_eff': d_rayleigh / pairwise_min_distance(centers_true),
                'snr_eff_nominal': 1.0/float(noise),
                'snr_eff_empirical': empirical_snr(ff_clean, ff_noisy),
                'mean_center_error': mean_err,
                'max_center_error': max_err,
                'resolved': bool(resolved),
                'rel_farfield_residual': rel_res,
                'true_centers': centers_true.tolist(),
                'init_centers': centers_init.tolist(),
                'rec_centers': centers_rec.tolist(),
            }
            rows.append(row)
            mode_dir = out_dir / f'noise_{noise:.2f}' / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            with open(mode_dir / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(row, f, indent=2)
            with open(mode_dir / 'history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            np.savez_compressed(mode_dir / 'reconstruction_result.npz', p_true=p_true, p_init=p_init, p_rec=p_rec, centers_true=centers_true)
            save_case_plot(mode_dir / 'reconstruction.png', p_true, p_init, p_rec,
                           title=(f'{mode}, noise={noise:.2f}\ntrue d_min={pairwise_min_distance(centers_true):.3f} ({pairwise_min_distance(centers_true)/d_rayleigh:.2f} d_R)\n'
                                  f'resolved={resolved}, d_rec_min={pairwise_min_distance(centers_rec):.3f}, err_max={max_err:.3f}'))
        np.savez_compressed(out_dir / f'music_image_noise_{noise:.2f}.npz', image=img, x_grid=x_grid, y_grid=y_grid)
        np.savez_compressed(out_dir / f'farfield_noisy_{noise:.2f}.npz', farfield_noisy=ff_noisy)

    with open(out_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'k': k, 'd_rayleigh': d_rayleigh, 'spacing_requested': float(args.spacing), 'metrics': rows}, f, indent=2)

    save_panel(out_dir / 'prior_sensitivity_panel.png', p_true, init_map, rec_map, modes, noises,
               pairwise_min_distance(centers_true), d_rayleigh)

    # simple bar/line plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for mode in modes:
        sel = [r for r in rows if r['init_mode']==mode]
        sel.sort(key=lambda z: z['noise'])
        axes[0].plot([r['noise'] for r in sel], [r['max_center_error'] for r in sel], marker='o', lw=2, label=mode)
        axes[1].plot([r['noise'] for r in sel], [1.0 if r['resolved'] else 0.0 for r in sel], marker='o', lw=2, label=mode)
    axes[0].set_xlabel('noise level'); axes[0].set_ylabel('max center error'); axes[0].grid(True, alpha=0.25)
    axes[0].set_title('Prior sensitivity: localization error')
    axes[1].set_xlabel('noise level'); axes[1].set_ylabel('resolved'); axes[1].set_ylim(-0.05, 1.05); axes[1].grid(True, alpha=0.25)
    axes[1].set_title('Prior sensitivity: resolved or not')
    axes[1].legend(loc='best')
    fig.savefig(out_dir / 'prior_sensitivity_metrics.png', dpi=180)
    plt.close(fig)

    print(json.dumps({
        'summary_csv': str(out_dir / 'summary.csv'),
        'summary_json': str(out_dir / 'summary.json'),
        'panel': str(out_dir / 'prior_sensitivity_panel.png'),
        'metrics_plot': str(out_dir / 'prior_sensitivity_metrics.png'),
    }, indent=2))

if __name__ == '__main__':
    main()
