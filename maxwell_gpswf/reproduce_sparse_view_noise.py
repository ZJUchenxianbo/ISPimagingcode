"""Independent Monte Carlo reproduction of the experiment described for Figure 6.2.

The original random samples, seed, tensor, noise amplitude, and trial count were
not available. Defaults below are declared reproduction settings, not recovered
original settings. Requires numpy and matplotlib. No VIE data are used.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def measurement_matrix(radius, count):
    """Columnwise tensor vectorization; two real orthonormal polarizations/pair."""
    t = np.sqrt(1.0 - radius**2)
    blocks = []
    for theta in 2 * np.pi * np.arange(count) / count:
        ct, st = np.cos(theta), np.sin(theta)
        d = np.array([t * ct, t * st, radius])
        xhat = np.array([t * ct, t * st, -radius])
        E = np.array([[-st, radius * ct], [ct, radius * st], [0.0, -t]])
        assert np.allclose(E.T @ E, np.eye(2), atol=1e-13)
        assert np.linalg.norm(E.T @ d) < 1e-13
        assert np.allclose(d - xhat, [0, 0, 2 * radius])
        blocks.append(np.kron(E.T, np.eye(3) - np.outer(xhat, xhat)))
    return np.vstack(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figure62_reproduction")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--sigma", type=float, default=1e-3,
                        help="Complex noise RMS per measured component; tensor norm is one.")
    parser.add_argument("--reference-csv", type=Path,
                        help="Optional original curves traced from the vector PDF, not raw samples.")
    args = parser.parse_args()
    if args.trials < 2 or args.sigma <= 0:
        parser.error("trials must be at least 2 and sigma must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    radii = np.linspace(0.04, 0.97, 38)
    tensor = np.array([[.20+.040j, .048j, -.026j],
                       [-.032j, .12+.030j, .026j],
                       [.034j, -.014j, .08+.025j]])
    tensor /= np.linalg.norm(tensor, "fro")
    c = tensor.reshape(-1, order="F")
    rng = np.random.default_rng(args.seed)
    errors = np.empty((2, len(radii), args.trials))
    singular = np.empty((2, len(radii), 9))
    rms_theory = np.empty((2, len(radii)))
    mse_zscores = np.empty((2, len(radii)))
    noise_energy = 0.0
    noise_count = 0
    noiseless_max = 0.0
    gram_subset_min = 0.0
    for ir, radius in enumerate(radii):
        matrices = [measurement_matrix(radius, J) for J in (3, 6)]
        # Six equiangular configurations contain the three at angles 0,120,240.
        difference = matrices[1].T @ matrices[1] - matrices[0].T @ matrices[0]
        gram_subset_min = min(gram_subset_min, float(np.linalg.eigvalsh(difference)[0]))
        for ij, M in enumerate(matrices):
            sv = np.linalg.svd(M, compute_uv=False)
            singular[ij, ir] = sv
            assert sv[-1] > 1e-10
            pinv = np.linalg.pinv(M)
            g = M @ c
            noiseless_max = max(noiseless_max, float(np.linalg.norm(pinv @ g - c)))
            noise = args.sigma / np.sqrt(2) * (
                rng.standard_normal((args.trials, len(g)))
                + 1j * rng.standard_normal((args.trials, len(g))))
            noisy = g[None, :] + noise
            recovered = noisy @ pinv.T  # Batch implementation of c_tilde=M^dagger g.
            relative_errors = np.linalg.norm(recovered - c[None, :], axis=1) / np.linalg.norm(c)
            errors[ij, ir] = relative_errors
            noise_energy += float(np.sum(np.abs(noise)**2))
            noise_count += noise.size
            mse = args.sigma**2 * np.sum(sv**-2) / np.linalg.norm(c)**2
            variance_squared_error = args.sigma**4 * np.sum(sv**-4) / np.linalg.norm(c)**4
            rms_theory[ij, ir] = np.sqrt(mse)
            mse_zscores[ij, ir] = (np.mean(relative_errors**2) - mse) / np.sqrt(
                variance_squared_error / args.trials)
    means = errors.mean(axis=2)
    standard_errors = errors.std(axis=2, ddof=1) / np.sqrt(args.trials)
    kappa = singular[:, :, 0] / singular[:, :, -1]
    assert noiseless_max < 1e-11
    assert gram_subset_min > -1e-11
    assert np.max(np.abs(mse_zscores)) < 6.0
    assert np.all(rms_theory[1] < rms_theory[0])
    np.savez_compressed(args.out / "reproduction_samples.npz", radii=radii,
                        tensor=tensor, errors=errors, means=means,
                        standard_errors=standard_errors, singular_values=singular,
                        theoretical_rms=rms_theory)
    with (args.out / "mean_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "mean_J3", "mean_J6", "se_J3", "se_J6", "rms_theory_J3", "rms_theory_J6"])
        writer.writerows(zip(radii, *means, *standard_errors, *rms_theory))
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    for ij, J in enumerate((3, 6)):
        ax.semilogy(radii, means[ij], label=rf"$J={J}$", linewidth=1.6)
    ax.set(xlim=(0, 1), xlabel=r"normalized Fourier radius $\rho=|\xi|/(2k)$",
           ylabel="mean relative tensor error")
    ax.grid(which="both", alpha=.2)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(args.out / "sparse_view_noise_reproduced.pdf")
    fig.savefig(args.out / "sparse_view_noise_reproduced.png", dpi=220)
    plt.close(fig)
    report = {
        "status": "independent reproduction; original random experiment settings unavailable",
        "model": "g=M c, three/six equiangular pairs, two orthonormal polarizations each",
        "tensor": "manuscript Q1 normalized to unit Frobenius norm",
        "seed": args.seed, "trials_per_radius_and_design": args.trials,
        "complex_noise_rms_per_component": args.sigma,
        "real_and_imaginary_standard_deviation": args.sigma / np.sqrt(2),
        "noise_normalization": "none; variance fixed per component, not fixed total relative noise",
        "radii": {"min": float(radii[0]), "max": float(radii[-1]), "count": len(radii)},
        "plot_statistic": "arithmetic mean of trialwise relative Frobenius errors, not RMS",
        "noiseless_recovery_max_error": noiseless_max,
        "empirical_noise_variance": noise_energy / noise_count,
        "max_mse_standard_error_units_from_exact_prediction": float(np.max(abs(mse_zscores))),
        "max_relative_condition_number_difference": float(np.max(abs(kappa[0] / kappa[1] - 1))),
        "max_gram_difference_negative_eigenvalue": gram_subset_min,
        "fraction_of_radii_with_smaller_J6_mean": float(np.mean(means[1] < means[0])),
        "median_J6_to_J3_mean_error_ratio": float(np.median(means[1] / means[0])),
        "rms_comparison": "theory used only for validation; displayed curves are unsmoothed Monte Carlo means",
    }
    if args.reference_csv:
        reference = np.genfromtxt(args.reference_csv, delimiter=",", names=True)
        report["original_curve_comparison"] = {}
        fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
        for ij, J in enumerate((3, 6)):
            color = f"C{ij}"
            original = reference[f"mean_J{J}"]
            x = reference["rho"]
            estimate = np.interp(x, radii, means[ij])
            report["original_curve_comparison"][f"J{J}"] = {
                "median_absolute_relative_difference": float(np.median(abs(estimate / original - 1))),
                "reference_kind": "curve coordinates extracted from original vector PDF, not simulation samples"}
            ax.semilogy(x, original, "--", color=color, alpha=.75,
                        label=rf"Original figure, $J={J}$")
            ax.semilogy(radii, means[ij], "-", color=color,
                        label=rf"Independent run, $J={J}$")
        ax.set(xlim=(0, 1), xlabel=r"normalized Fourier radius $\rho=|\xi|/(2k)$",
               ylabel="mean relative tensor error")
        ax.grid(which="both", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(args.out / "comparison_with_original.png", dpi=220)
        plt.close(fig)
    (args.out / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
