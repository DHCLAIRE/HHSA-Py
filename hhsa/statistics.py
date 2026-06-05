"""Statistics helpers for HHT/HHSA outputs."""

from __future__ import annotations

import numpy as np


def mode_energy(modes: np.ndarray) -> np.ndarray:
    """Return sum-of-squares energy for each IMF."""

    arr = np.asarray(modes, dtype=float)
    if arr.ndim != 2:
        raise ValueError("modes must have shape (n_modes, n_samples)")
    return np.sum(arr**2, axis=1)


def normalized_entropy(values: np.ndarray) -> float:
    """Return Shannon entropy normalized to [0, 1]."""

    x = np.asarray(values, dtype=float)
    x = x[x > 0]
    if x.size <= 1:
        return 0.0
    p = x / np.sum(x)
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def orthogonality_index(modes: np.ndarray, signal: np.ndarray) -> float:
    """Estimate IMF orthogonality; lower values indicate cleaner separation."""

    arr = np.asarray(modes, dtype=float)
    x = np.asarray(signal, dtype=float)
    denom = np.sum(x**2) + np.finfo(float).eps
    cross = 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            if i != j:
                cross += abs(np.sum(arr[i] * arr[j]))
    return float(cross / denom)


def marginal_spectrum(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    *,
    bins: int = 128,
    freq_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple Hilbert marginal spectrum."""

    freq = np.asarray(frequency, dtype=float).ravel()
    amp = np.asarray(amplitude, dtype=float).ravel()
    mask = np.isfinite(freq) & np.isfinite(amp) & (freq >= 0)
    if freq_range is not None:
        mask &= (freq >= freq_range[0]) & (freq <= freq_range[1])
    hist, edges = np.histogram(freq[mask], bins=bins, range=freq_range, weights=amp[mask] ** 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
