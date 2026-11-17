"""Replot conditioning using analytic Gram matrices and archived experiment data."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def singular_values(radius, count):
    gram = np.zeros((9, 9))
    for angle in 2 * np.pi * np.arange(count) / count:
        direction = np.array([np.sqrt(1-radius**2)*np.cos(angle),
                              np.sqrt(1-radius**2)*np.sin(angle), radius])
        observation = direction.copy()
        observation[2] = -radius
        pd = np.eye(3) - np.outer(direction, direction)
        px = np.eye(3) - np.outer(observation, observation)
        gram += np.kron(pd, px)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0))


def plot_panel(ax, radii, curves, measured_radii, measured, kind):
    for count, values in curves.items():
        y = values[:, 0] if kind == 'sigma' else values[:, -1]/values[:, 0]
        ax.plot(radii, y, label=f'equiangular $J={count}$', linewidth=1.6,
                linestyle='--' if kind == 'kappa' and count == 6 else '-')
    ax.scatter(measured_radii, measured, s=8, alpha=.45,
               label='selected reconstruction geometry', rasterized=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'normalized Fourier radius $\rho=|\xi|/(2k)$')
    ax.set_ylabel(r'$\sigma_{\min}(M)$' if kind == 'sigma' else r'$\kappa_2(M)$')
    if kind == 'kappa':
        ax.set_yscale('log')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(
        r'C:\Users\123\Desktop\Imagingcode\project\ISPimagingcode\outputs\exp6\exp6_tensor_diagnostics_detail.npz'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as archive:
        measured_radii = np.linalg.norm(archive['target_nodes'], axis=1)
        sigma = archive['polarimetric_sigma_min'].copy()
        kappa = archive['polarimetric_condition_numbers'].copy()
    radii = np.linspace(.005, .995, 400)
    curves = {count: np.array([singular_values(r, count) for r in radii])
              for count in (3, 6)}
    for kind, measured, filename in [('sigma', sigma, 'conditioning_sigma_min'),
                                      ('kappa', kappa, 'conditioning_kappa')]:
        fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
        plot_panel(ax, radii, curves, measured_radii, measured, kind)
        fig.savefig(args.out / (filename + '.pdf'))
        fig.savefig(args.out / (filename + '.png'), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1), constrained_layout=True)
    plot_panel(axes[0], radii, curves, measured_radii, sigma, 'sigma')
    plot_panel(axes[1], radii, curves, measured_radii, kappa, 'kappa')
    fig.savefig(args.out / 'conditioning_preview.png', dpi=160)
    plt.close(fig)
    print(f'Replotted {len(measured_radii)} archived configurations.')
    print(f'sigma minimum={sigma.min():.6f}, median={np.median(sigma):.6f}')
    print(f'kappa median={np.median(kappa):.6f}, maximum={kappa.max():.6f}')


if __name__ == '__main__':
    main()
