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
    envelope_mean_tol: float = 0.1
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


def _is_terminal_residue(x: np.ndarray) -> bool:
    """Return True when a residue cannot produce another oscillatory IMF."""

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


def _zero_crossing_count(x: np.ndarray) -> int:
    """Count sign changes through zero, ignoring zero-only touches."""

    signs = np.sign(x)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))


def _has_small_envelope_mean(candidate: np.ndarray, *, tolerance: float) -> bool:
    mean = _mean_envelope(candidate)
    scale = np.linalg.norm(candidate) + np.finfo(float).eps
    return bool(np.linalg.norm(mean) / scale <= tolerance)


def _is_imf(candidate: np.ndarray, *, envelope_mean_tol: float = 0.1) -> bool:
    """Check both IMF conditions used by the EMD sifting stop rule."""

    maxima, minima = _extrema_indices(candidate)
    extrema_count = maxima.size + minima.size
    zero_crossings = _zero_crossing_count(candidate)
    has_balanced_events = abs(extrema_count - zero_crossings) <= 1
    return has_balanced_events and _has_small_envelope_mean(candidate, tolerance=envelope_mean_tol)


def _sift_first_imf(signal: np.ndarray, config: EMDConfig) -> np.ndarray:
    h = signal.copy()
    eps = np.finfo(float).eps
    for _ in range(config.max_siftings):
        if _is_terminal_residue(h):
            break
        previous = h.copy()
        h = h - _mean_envelope(h)
        sd = np.sum((previous - h) ** 2) / (np.sum(previous**2) + eps)
        if sd < config.stop_sd and _is_imf(h, envelope_mean_tol=config.envelope_mean_tol):
            break
    return h


def emd(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
    envelope_mean_tol: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal into IMFs and a residue with vanilla EMD."""

    x = _as_1d(signal)
    config = EMDConfig(
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        envelope_mean_tol=envelope_mean_tol,
    )
    residue = x.copy()
    imfs: list[np.ndarray] = []

    while not _is_terminal_residue(residue):
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
    envelope_mean_tol: float = 0.1,
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

    while not _is_terminal_residue(residue):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break
        members = []
        for _ in range(ensemble_size):
            noise = _normalize_noise(rng.normal(size=x.size))
            noisy = residue + scale * noise
            first, _ = emd(
                noisy,
                max_imfs=1,
                max_siftings=max_siftings,
                stop_sd=stop_sd,
                envelope_mean_tol=envelope_mean_tol,
            )
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
    envelope_mean_tol: float = 0.1,
    snr_flag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved CEEMDAN-style decomposition.

    This follows the Colominas-style MATLAB code structure:
    pre-decompose each white-noise realization, estimate the first local mean
    from noisy copies of the normalized signal, then obtain every next mode as
    the difference between consecutive averaged local means.
    """

    x = _as_1d(signal)
    if snr_flag not in {1, 2}:
        raise ValueError("snr_flag must be 1 or 2")
    x_std = np.std(x)
    if x_std == 0:
        return np.empty((0, x.size)), x.copy()
    x_norm = x / x_std
    rng = np.random.default_rng(random_state)
    noise_modes: list[np.ndarray] = []
    noise_max_imfs = None if max_imfs is None else max_imfs + 1
    for _ in range(ensemble_size):
        noise = rng.normal(size=x.size)
        modes, residue = emd(
            noise,
            max_imfs=noise_max_imfs,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            envelope_mean_tol=envelope_mean_tol,
        )
        noise_modes.append(np.vstack((modes, residue[np.newaxis, :])))

    first_means = []
    for modes in noise_modes:
        noise = _normalize_noise(modes[0])
        noisy_signal = x_norm + noise_width * noise
        first, noisy_residue = emd(
            noisy_signal,
            max_imfs=1,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            envelope_mean_tol=envelope_mean_tol,
        )
        first_means.append(noisy_residue if first.size else noisy_signal)

    current_mean = np.mean(first_means, axis=0)
    imfs: list[np.ndarray] = [x_norm - current_mean]
    mode_index = 1

    while not _is_terminal_residue(current_mean):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break

        next_means = []
        for modes in noise_modes:
            if modes.shape[0] > mode_index:
                noise = modes[mode_index]
                if snr_flag == 2:
                    noise = _normalize_noise(noise)
                noisy_mean = current_mean + np.std(current_mean) * noise_width * noise
            else:
                noisy_mean = current_mean

            _, noisy_residue = emd(
                noisy_mean,
                max_imfs=1,
                max_siftings=max_siftings,
                stop_sd=stop_sd,
                envelope_mean_tol=envelope_mean_tol,
            )
            next_means.append(noisy_residue)

        next_mean = np.mean(next_means, axis=0)
        imf = current_mean - next_mean
        if np.allclose(imf, 0):
            break
        imfs.append(imf)
        current_mean = next_mean
        mode_index += 1

    modes = np.vstack(imfs) * x_std
    residue = current_mean * x_std
    return modes, residue

