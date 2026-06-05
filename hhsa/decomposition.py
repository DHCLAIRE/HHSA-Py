"""EMD, CEEMDAN, and ICEEMDAN-style decomposition.

The implementations here are intentionally compact and dependency-light so the
pipeline is easy to study. If PyEMD or emd is available in your environment, you
can swap this module for those battle-tested decomposers while keeping the HHSA
pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import argrelextrema


@dataclass(frozen=True)
class EMDConfig:
    max_imfs: int | None = None
    max_siftings: int = 50
    stop_sd: float = 0.2
    extrema_padding: int = 2


def _as_1d(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 4:
        raise ValueError("signal must contain at least four samples")
    return x


def _extrema_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxima = argrelextrema(x, np.greater)[0]
    minima = argrelextrema(x, np.less)[0]
    return maxima, minima


def _is_monotonic_residue(x: np.ndarray) -> bool:
    maxima, minima = _extrema_indices(x)
    return maxima.size + minima.size < 2


def _pad_extrema(indices: np.ndarray, n: int) -> np.ndarray:
    if indices.size == 0:
        return np.array([0, n - 1])
    padded = np.unique(np.concatenate(([0], indices, [n - 1]))).astype(int)
    return padded


def _envelope(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = x.size
    t = np.arange(n)
    knots = _pad_extrema(indices, n)
    values = x[knots]
    if knots.size >= 4:
        return CubicSpline(knots, values, bc_type="natural")(t)
    return interp1d(knots, values, kind="linear", fill_value="extrapolate")(t)


def _mean_envelope(x: np.ndarray) -> np.ndarray:
    maxima, minima = _extrema_indices(x)
    upper = _envelope(x, maxima)
    lower = _envelope(x, minima)
    return 0.5 * (upper + lower)


def _is_imf(candidate: np.ndarray) -> bool:
    maxima, minima = _extrema_indices(candidate)
    extrema_count = maxima.size + minima.size
    signs = np.signbit(candidate)
    zero_crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return abs(extrema_count - zero_crossings) <= 1


def _sift_first_imf(signal: np.ndarray, config: EMDConfig) -> np.ndarray:
    h = signal.copy()
    eps = np.finfo(float).eps
    for _ in range(config.max_siftings):
        if _is_monotonic_residue(h):
            break
        previous = h.copy()
        h = h - _mean_envelope(h)
        sd = np.sum((previous - h) ** 2) / (np.sum(previous**2) + eps)
        if sd < config.stop_sd and _is_imf(h):
            break
    return h


def emd(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal into IMFs and a residue with vanilla EMD."""

    x = _as_1d(signal)
    config = EMDConfig(max_imfs=max_imfs, max_siftings=max_siftings, stop_sd=stop_sd)
    residue = x.copy()
    imfs: list[np.ndarray] = []

    while not _is_monotonic_residue(residue):
        if config.max_imfs is not None and len(imfs) >= config.max_imfs:
            break
        imf = _sift_first_imf(residue, config)
        if np.allclose(imf, 0):
            break
        imfs.append(imf)
        residue = residue - imf

    if not imfs:
        return np.empty((0, x.size)), residue
    return np.vstack(imfs), residue


def _normalize_noise(noise: np.ndarray) -> np.ndarray:
    std = np.std(noise)
    if std == 0:
        return noise
    return noise / std


def ceemdan(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | np.random.Generator | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete ensemble EMD with adaptive noise.

    This educational implementation estimates each next IMF by averaging the
    first IMF extracted from noisy copies of the current residue.
    """

    x = _as_1d(signal)
    rng = np.random.default_rng(random_state)
    residue = x.copy()
    imfs: list[np.ndarray] = []
    scale = noise_width * np.std(x)

    while not _is_monotonic_residue(residue):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break
        members = []
        for _ in range(ensemble_size):
            noise = _normalize_noise(rng.normal(size=x.size))
            noisy = residue + scale * noise
            first, _ = emd(noisy, max_imfs=1, max_siftings=max_siftings, stop_sd=stop_sd)
            if first.size:
                members.append(first[0])
        if not members:
            break
        imf = np.mean(members, axis=0)
        imfs.append(imf)
        residue = residue - imf

    if not imfs:
        return np.empty((0, x.size)), residue
    return np.vstack(imfs), residue


def iceemdan(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | np.random.Generator | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved CEEMDAN-style decomposition.

    ICEEMDAN obtains each mode from the difference between consecutive averaged
    local means of noisy residue realizations. This compact version follows that
    idea while reusing the local mean implied by one-IMF EMD.
    """

    x = _as_1d(signal)
    rng = np.random.default_rng(random_state)
    residue = x.copy()
    imfs: list[np.ndarray] = []
    scale = noise_width * np.std(x)

    while not _is_monotonic_residue(residue):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break

        local_means = []
        for _ in range(ensemble_size):
            noise = _normalize_noise(rng.normal(size=x.size))
            noisy = residue + scale * noise
            first, noisy_residue = emd(noisy, max_imfs=1, max_siftings=max_siftings, stop_sd=stop_sd)
            local_means.append(noisy_residue if first.size else noisy)

        next_residue = np.mean(local_means, axis=0)
        imf = residue - next_residue
        if np.allclose(imf, 0):
            break
        imfs.append(imf)
        residue = next_residue

    if not imfs:
        return np.empty((0, x.size)), residue
    return np.vstack(imfs), residue
