#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions: table formatting, vector norms, IO helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


class SimpleColumn:
    """Small ndarray wrapper used when pandas is unavailable."""

    def __init__(self, values: list[object]):
        self.values = np.asarray(values)

    def to_numpy(self, dtype=None) -> NDArray:
        return self.values.astype(dtype) if dtype is not None else self.values

    def __array__(self, dtype=None) -> NDArray:
        return self.to_numpy(dtype=dtype)

    def __eq__(self, other) -> NDArray:  # type: ignore[override]
        return self.values == other

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


class SimpleTable:
    """Minimal table fallback with the subset of DataFrame behavior used here."""

    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def __getitem__(self, key):
        if isinstance(key, str):
            return SimpleColumn([row[key] for row in self.rows])
        mask = np.asarray(key, dtype=bool)
        return SimpleTable([row for row, keep in zip(self.rows, mask) if bool(keep)])

    def iterrows(self):
        for idx, row in enumerate(self.rows):
            yield idx, row

    def to_csv(self, path: Path, index: bool = False) -> None:
        del index
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def to_string(self, index: bool = False, float_format=None) -> str:
        del index
        rendered_rows = []
        for row in self.rows:
            rendered = {}
            for key in self.columns:
                value = row[key]
                if isinstance(value, float) and float_format is not None:
                    rendered[key] = float_format(value)
                else:
                    rendered[key] = str(value)
            rendered_rows.append(rendered)
        widths = {
            key: max(len(key), *(len(row[key]) for row in rendered_rows))
            for key in self.columns
        }
        lines = [" ".join(key.rjust(widths[key]) for key in self.columns)]
        for row in rendered_rows:
            lines.append(" ".join(row[key].rjust(widths[key]) for key in self.columns))
        return "\n".join(lines)


def make_table(rows: list[dict[str, object]]) -> Any:
    """Return a pandas DataFrame when available, otherwise a lightweight table."""
    if pd is not None:
        return pd.DataFrame(rows)
    return SimpleTable(rows)


def vector_norm(x: NDArray) -> float:
    """Return the Euclidean norm as a Python float."""
    return float(np.linalg.norm(x))


def print_table(title: str, df: Any) -> None:
    """Print a table with compact scientific notation."""
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))


def complex_relative_noise(g: CArray, delta: float, rng: np.random.Generator) -> CArray:
    """Complex Gaussian noise with norm ``delta * ||g||``."""
    eta = rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape)
    eta_norm = max(vector_norm(eta), 1e-14)
    return eta / eta_norm * float(delta) * vector_norm(g)
