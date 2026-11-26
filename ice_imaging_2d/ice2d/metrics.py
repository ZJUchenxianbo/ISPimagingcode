"""Truth is accessed only here and in figures after imaging has finished."""

import numpy as np


def extract_ridge(image, z):
    image, z = np.asarray(image), np.asarray(z)
    if (image.ndim != 2 or image.shape[0] != len(z) or not np.isfinite(image).all()
            or not len(z) or np.any(np.max(image, axis=0) <= 0)):
        raise ValueError('Cannot extract a boundary from invalid or unilluminated indicator columns')
    index = np.argmax(image, axis=0)
    ridge = z[index].copy()
    # Subpixel quadratic peak interpolation, no horizontal smoothing or truth.
    for column, row in enumerate(index):
        if 0 < row < len(z) - 1:
            left, centre, right = image[row - 1:row + 2, column]
            curvature = left - 2 * centre + right
            if curvature < 0:
                offset = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)
                ridge[column] += offset * (z[1] - z[0])
    return ridge, float(np.mean((index == 0) | (index == len(z) - 1)))


def evaluate(image, x, z, truth, reference_bottom):
    ridge, edge_fraction = extract_ridge(image, z)
    error = ridge - truth
    return dict(rmse_m=float(np.sqrt(np.mean(error ** 2))),
                maximum_error_m=float(np.max(abs(error))),
                bias_m=float(np.mean(error)),
                peak_to_valley_m=float(np.ptp(ridge)),
                true_peak_to_valley_m=float(np.ptp(truth)),
                crest_x_m=float(x[np.argmax(ridge)]), trough_x_m=float(x[np.argmin(ridge)]),
                boundary_fraction=edge_fraction,
                constant_depth_rmse_m=float(np.sqrt(np.mean((reference_bottom - truth) ** 2))))
