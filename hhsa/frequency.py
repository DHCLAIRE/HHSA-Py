"""Instantaneous-frequency estimators for HHT/HHSA."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert


def _sample_rate_to_dt(sample_rate: float) -> float:
    """Validate a positive sample rate and return its sampling interval."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return 1.0 / sample_rate


def _zero_crossing_times(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Estimate sub-sample zero-crossing times by linear interpolation."""

    signs = np.signbit(x)
    crossings = np.flatnonzero(signs[1:] != signs[:-1])
    times = []
    for i in crossings:
        y0, y1 = x[i], x[i + 1]
        frac = 0.0 if y1 == y0 else abs(y0) / (abs(y0) + abs(y1))
        times.append((i + frac) / sample_rate)
    return np.asarray(times)


def _extrema_times(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return times of local turning points estimated from derivative sign changes."""

    dx = np.diff(x)
    turning = np.flatnonzero(np.signbit(dx[1:]) != np.signbit(dx[:-1])) + 1
    return turning / sample_rate


def _interp_frequency(event_times: np.ndarray, n: int, sample_rate: float, period_multiplier: float) -> np.ndarray:
    """Interpolate instantaneous frequency from periodic event timings."""

    if event_times.size < 2:
        return np.full(n, np.nan)
    dt = np.diff(event_times)
    valid = dt > 0
    if not np.any(valid):
        return np.full(n, np.nan)
    centers = 0.5 * (event_times[:-1] + event_times[1:])[valid]
    freq = 1.0 / (dt[valid] * period_multiplier)
    t = np.arange(n) / sample_rate
    return interp1d(centers, freq, bounds_error=False, fill_value=(freq[0], freq[-1]))(t)


def generalized_zero_crossing(imf: np.ndarray, sample_rate: float) -> np.ndarray:
    """Estimate instantaneous frequency using generalized zero crossing.

    The estimate combines zero-crossing half-periods, extrema half-periods, and
    quarter-period spacing from the union of both event types. Their median is
    robust for short, noisy IMFs.
    """

    x = np.asarray(imf, dtype=float)
    zc = _zero_crossing_times(x, sample_rate)
    extrema = _extrema_times(x, sample_rate)
    all_events = np.sort(np.unique(np.concatenate((zc, extrema))))

    candidates = np.vstack(
        [
            _interp_frequency(zc, x.size, sample_rate, period_multiplier=2.0),
            _interp_frequency(extrema, x.size, sample_rate, period_multiplier=2.0),
            _interp_frequency(all_events, x.size, sample_rate, period_multiplier=4.0),
        ]
    )
    with np.errstate(invalid="ignore"):
        freq = np.nanmedian(candidates, axis=0)
    return np.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)


def quadrature_frequency(imf: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate amplitude, phase, and instantaneous frequency by quadrature.

    The quadrature component is obtained from the Hilbert transform. Frequency is
    the derivative of the unwrapped analytic phase.
    """

    _sample_rate_to_dt(sample_rate)
    x = np.asarray(imf, dtype=float)
    analytic = hilbert(x)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    frequency = np.gradient(phase) * sample_rate / (2.0 * np.pi)
    frequency = np.maximum(frequency, 0.0)
    return amplitude, phase, frequency


def frequency_transform(
    imfs: np.ndarray,
    sample_rate: float,
    *,
    method: str = "quad",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return instantaneous amplitude, phase, and frequency for every IMF."""

    modes = np.asarray(imfs, dtype=float)
    if modes.ndim != 2:
        raise ValueError("imfs must have shape (n_modes, n_samples)")

    amplitudes = []
    phases = []
    frequencies = []
    for imf in modes:
        amp, phase, quad_freq = quadrature_frequency(imf, sample_rate)
        if method.lower() == "quad":
            freq = quad_freq
        elif method.lower() == "gzc":
            freq = generalized_zero_crossing(imf, sample_rate)
        elif method.lower() in {"hybrid", "gzc_quad", "quad_gzc"}:
            gzc = generalized_zero_crossing(imf, sample_rate)
            freq = np.where(gzc > 0, 0.5 * (gzc + quad_freq), quad_freq)
        else:
            raise ValueError("method must be 'quad', 'gzc', or 'hybrid'")
        amplitudes.append(amp)
        phases.append(phase)
        frequencies.append(freq)

    return np.vstack(amplitudes), np.vstack(phases), np.vstack(frequencies)
