#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared U-Net utilities for learning-enhanced sampling images."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None

from common.scattering import Array, add_relative_noise
from common.forward import solve_point_scatterer_farfield
from common.sampling import (
    aperture_angles,
    aperture_measure,
    direct_sampling_indicator,
    normalize_indicator,
)

CArray = NDArray[np.complex128]


def _require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "PyTorch is required for U-Net training. Install it in this project environment with "
            r".\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


if nn is not None:

    class DoubleConv(nn.Module):
        """Two padded convolutions used in each U-Net stage."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):  # noqa: ANN001
            return self.net(x)


    class UNet2D(nn.Module):
        """Small 2D U-Net for single-channel imaging indicators."""

        def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 8) -> None:
            super().__init__()
            self.enc1 = DoubleConv(in_channels, base_channels)
            self.enc2 = DoubleConv(base_channels, base_channels * 2)
            self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base_channels * 2, base_channels)
            self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        def forward(self, x):  # noqa: ANN001
            x1 = self.enc1(x)
            x2 = self.enc2(self.pool(x1))
            xb = self.bottleneck(self.pool(x2))

            u2 = self.up2(xb)
            if u2.shape[-2:] != x2.shape[-2:]:
                u2 = F.interpolate(u2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat([u2, x2], dim=1))

            u1 = self.up1(d2)
            if u1.shape[-2:] != x1.shape[-2:]:
                u1 = F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([u1, x1], dim=1))
            return torch.sigmoid(self.out_conv(d1))

else:

    class UNet2D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            _require_torch()


@dataclass(frozen=True)
class PointScattererUNetConfig:
    """Synthetic training setup for DSM-to-support U-Net experiments."""

    k: float
    incident_angles: Array
    aperture_center: float
    aperture_half_widths: tuple[float, ...]
    noise_level: float
    grid_extent: float
    grid_size: int = 96
    n_obs: int = 61
    n_samples: int = 64
    max_points: int = 3
    target_sigma: float = 0.035
    block_size: int = 8192
    seed: int = 20260518


def point_support_heatmap(points: Array, x_grid: Array, y_grid: Array, sigma: float) -> Array:
    """Build a normalized Gaussian support image from point scatterer locations."""
    points = np.asarray(points, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if points.size == 0:
        return np.zeros((y_grid.size, x_grid.size), dtype=float)

    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    target = np.zeros_like(X, dtype=float)
    sigma2 = max(float(sigma) ** 2, 1e-12)
    for px, py in points:
        target = np.maximum(target, np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2.0 * sigma2)))
    return normalize_indicator(target)


def _normalize_minmax(image: Array) -> Array:
    image = np.asarray(image, dtype=float)
    lo = float(np.min(image))
    hi = float(np.max(image))
    if hi - lo <= 1e-14:
        return np.zeros_like(image, dtype=float)
    return (image - lo) / (hi - lo)


def _sample_points(
    rng: np.random.Generator,
    n_points: int,
    extent: float,
    min_separation: float,
) -> Array:
    points: list[NDArray[np.float64]] = []
    radius = 0.78 * float(extent)
    for _ in range(400):
        candidate = rng.uniform(-radius, radius, size=2)
        if all(float(np.linalg.norm(candidate - point)) >= min_separation for point in points):
            points.append(candidate.astype(float))
            if len(points) == n_points:
                break
    while len(points) < n_points:
        points.append(rng.uniform(-radius, radius, size=2).astype(float))
    return np.asarray(points, dtype=float)


def build_point_scatterer_unet_dataset(config: PointScattererUNetConfig) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate synthetic pairs (DSM indicator, point-support heatmap)."""
    rng = np.random.default_rng(config.seed)
    x_grid = np.linspace(-float(config.grid_extent), float(config.grid_extent), int(config.grid_size))
    y_grid = np.linspace(-float(config.grid_extent), float(config.grid_extent), int(config.grid_size))
    inputs = np.empty((config.n_samples, 1, config.grid_size, config.grid_size), dtype=np.float32)
    targets = np.empty_like(inputs)

    half_widths = tuple(float(alpha) for alpha in config.aperture_half_widths)
    min_sep = max(0.08 * float(config.grid_extent), 2.0 * float(config.target_sigma))
    for sample_idx in range(int(config.n_samples)):
        n_points = int(rng.integers(1, int(config.max_points) + 1))
        points = _sample_points(rng, n_points, float(config.grid_extent), min_sep)
        magnitudes = rng.uniform(0.75, 1.25, size=n_points)
        phases = rng.uniform(-math.pi, math.pi, size=n_points)
        strengths = (magnitudes * np.exp(1j * phases)).astype(np.complex128)

        alpha = half_widths[int(rng.integers(0, len(half_widths)))]
        obs_angles = aperture_angles(float(config.aperture_center), alpha, int(config.n_obs))
        farfield = solve_point_scatterer_farfield(points, strengths, float(config.k), config.incident_angles, obs_angles)
        rel_noise = rng.uniform(0.0, float(config.noise_level))
        noisy_farfield = add_relative_noise(farfield, rel_noise, rng)
        indicator = direct_sampling_indicator(
            noisy_farfield,
            float(config.k),
            obs_angles,
            config.incident_angles,
            x_grid,
            y_grid,
            aperture_measure(alpha),
            block_size=int(config.block_size),
        )
        target = point_support_heatmap(points, x_grid, y_grid, float(config.target_sigma))
        inputs[sample_idx, 0] = indicator.astype(np.float32)
        targets[sample_idx, 0] = target.astype(np.float32)

    return inputs, targets


def train_unet_from_arrays(
    inputs: NDArray[np.float32],
    targets: NDArray[np.float32],
    *,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    base_channels: int = 8,
    seed: int = 20260518,
    device: str | None = None,
) -> tuple[UNet2D, list[float]]:
    """Train a U-Net on already built image pairs."""
    _require_torch()
    if inputs.shape != targets.shape:
        raise ValueError(f"inputs and targets must have the same shape, got {inputs.shape} and {targets.shape}")
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError("inputs must have shape (n_samples, 1, height, width)")

    torch.manual_seed(int(seed))
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNet2D(in_channels=1, out_channels=1, base_channels=int(base_channels)).to(torch_device)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    history: list[float] = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs)
            point_weight = 1.0 + 6.0 * batch_targets
            loss = torch.mean(point_weight * (prediction - batch_targets) ** 2)
            loss.backward()
            optimizer.step()
            batch_size_now = int(batch_inputs.shape[0])
            total_loss += float(loss.item()) * batch_size_now
            seen += batch_size_now
        epoch_loss = total_loss / max(seen, 1)
        history.append(epoch_loss)
        print(f"unet epoch {epoch + 1}/{int(epochs)} loss={epoch_loss:.6f}", flush=True)

    return model, history


def train_point_scatterer_unet(
    config: PointScattererUNetConfig,
    *,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    base_channels: int = 8,
    device: str | None = None,
) -> tuple[UNet2D, list[float]]:
    """Generate synthetic point-scatterer data and train the U-Net."""
    inputs, targets = build_point_scatterer_unet_dataset(config)
    return train_unet_from_arrays(
        inputs,
        targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_channels=base_channels,
        seed=config.seed,
        device=device,
    )


def predict_unet_images(
    model: UNet2D,
    images: Iterable[Array],
    *,
    network_size: int | None = None,
    batch_size: int = 4,
    device: str | None = None,
    normalize: bool = True,
) -> list[Array]:
    """Apply a trained U-Net to a list of 2D indicators."""
    _require_torch()
    image_list = [np.asarray(image, dtype=np.float32) for image in images]
    if not image_list:
        return []
    original_shape = image_list[0].shape
    if any(image.shape != original_shape for image in image_list):
        raise ValueError("all images must have the same shape")

    data = np.stack(image_list, axis=0)[:, None, :, :]
    torch_device = torch.device(device or next(model.parameters()).device)
    tensor = torch.from_numpy(data).to(torch_device)
    output_chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor.shape[0], int(batch_size)):
            chunk = tensor[start : start + int(batch_size)]
            if network_size is not None and tuple(chunk.shape[-2:]) != (int(network_size), int(network_size)):
                chunk = F.interpolate(chunk, size=(int(network_size), int(network_size)), mode="bilinear", align_corners=False)
                prediction = model(chunk)
                prediction = F.interpolate(prediction, size=original_shape, mode="bilinear", align_corners=False)
            else:
                prediction = model(chunk)
            output_chunks.append(prediction.detach().cpu().numpy())

    outputs = np.concatenate(output_chunks, axis=0)[:, 0]
    if normalize:
        return [_normalize_minmax(output.astype(float)) for output in outputs]
    return [output.astype(float) for output in outputs]


def save_unet_checkpoint(path: str | Path, model: UNet2D, history: list[float], metadata: dict[str, object]) -> None:
    """Save a trained U-Net checkpoint with minimal experiment metadata."""
    _require_torch()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "metadata": metadata,
        },
        Path(path),
    )
