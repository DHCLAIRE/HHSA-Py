"""Statistics helpers for HHT/HHSA outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalTestResult:
    """Container for feature-wise HHSA statistical test results."""

    statistic: np.ndarray
    pvalue: np.ndarray
    feature: str
    method: str
    n_group_a: int
    n_group_b: int


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


# Histogram definition: low frequency, high frequency, bin count, and scale.
SpectrumBins = tuple[float, float, int, str]


def spectrum_bin_edges(hist: SpectrumBins) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram centers and edges from an EMD-style bin definition."""

    low, high, count, scale = hist
    if low <= 0 and scale == "log":
        raise ValueError("log-spaced bins require a positive lower bound")
    if high <= low:
        raise ValueError("histogram upper bound must be greater than lower bound")
    if count < 1:
        raise ValueError("histogram bin count must be positive")
    if scale == "log":
        edges = np.geomspace(low, high, count + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    elif scale == "linear":
        edges = np.linspace(low, high, count + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        raise ValueError("histogram scale must be 'linear' or 'log'")
    return centers, edges


def hilbert_huang_spectrum(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    hist: SpectrumBins,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute marginal and time-resolved Hilbert-Huang spectra.

    Parameters use the package convention ``(n_modes, n_samples)``. Power is
    accumulated as squared instantaneous amplitude.
    """

    freq = np.asarray(frequency, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    if freq.shape != amp.shape or freq.ndim != 2:
        raise ValueError("frequency and amplitude must both have shape (n_modes, n_samples)")

    centers, edges = spectrum_bin_edges(hist)
    time_frequency = np.zeros((centers.size, freq.shape[1]))
    for mode_freq, mode_amp in zip(freq, amp):
        for sample_index, (f, a) in enumerate(zip(mode_freq, mode_amp)):
            if np.isfinite(f) and np.isfinite(a) and edges[0] <= f <= edges[-1]:
                bin_index = np.searchsorted(edges, f, side="right") - 1
                bin_index = min(max(bin_index, 0), centers.size - 1)
                time_frequency[bin_index, sample_index] += a**2

    marginal = time_frequency.sum(axis=1)
    return centers, marginal, time_frequency


def holospectrum(
    carrier_frequency: np.ndarray,
    am_frequency: list[np.ndarray],
    am_amplitude: list[np.ndarray],
    carrier_hist: SpectrumBins,
    am_hist: SpectrumBins,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a time-averaged Holo-Hilbert spectrum.

    ``carrier_frequency`` has shape ``(n_carrier_modes, n_samples)``. Each
    second-layer entry in ``am_frequency`` and ``am_amplitude`` has shape
    ``(n_am_modes, n_samples)`` for the corresponding carrier IMF.
    """

    carrier = np.asarray(carrier_frequency, dtype=float)
    if carrier.ndim != 2:
        raise ValueError("carrier_frequency must have shape (n_modes, n_samples)")
    if len(am_frequency) != carrier.shape[0] or len(am_amplitude) != carrier.shape[0]:
        raise ValueError("second-layer lists must have one entry per carrier IMF")

    carrier_centers, carrier_edges = spectrum_bin_edges(carrier_hist)
    am_centers, am_edges = spectrum_bin_edges(am_hist)
    holo = np.zeros((carrier_centers.size, am_centers.size))

    for mode_index, carrier_freq in enumerate(carrier):
        am_freq = np.asarray(am_frequency[mode_index], dtype=float)
        am_amp = np.asarray(am_amplitude[mode_index], dtype=float)
        if am_freq.size == 0 or am_amp.size == 0:
            continue
        if am_freq.shape != am_amp.shape or am_freq.ndim != 2:
            raise ValueError("each second-layer frequency/amplitude pair must have shape (n_am_modes, n_samples)")
        if am_freq.shape[1] != carrier.shape[1]:
            raise ValueError("second-layer sample count must match carrier_frequency")

        for layer_freq, layer_amp in zip(am_freq, am_amp):
            for sample_index, (am_f, am_a) in enumerate(zip(layer_freq, layer_amp)):
                carrier_f = carrier_freq[sample_index]
                if not (np.isfinite(carrier_f) and np.isfinite(am_f) and np.isfinite(am_a)):
                    continue
                if not (carrier_edges[0] <= carrier_f <= carrier_edges[-1]):
                    continue
                if not (am_edges[0] <= am_f <= am_edges[-1]):
                    continue
                carrier_bin = np.searchsorted(carrier_edges, carrier_f, side="right") - 1
                am_bin = np.searchsorted(am_edges, am_f, side="right") - 1
                carrier_bin = min(max(carrier_bin, 0), carrier_centers.size - 1)
                am_bin = min(max(am_bin, 0), am_centers.size - 1)
                holo[carrier_bin, am_bin] += am_a**2

    return carrier_centers, am_centers, holo


def hhsa_feature(result: object, feature: str = "mode_energy") -> np.ndarray:
    """Extract a flattened statistical feature vector from one HHSA result."""

    if feature == "mode_energy":
        return mode_energy(result.imfs)
    if feature == "marginal":
        return np.asarray(result.marginal, dtype=float).ravel()
    if feature == "hht":
        return np.asarray(result.hht, dtype=float).ravel()
    if feature == "holospectrum":
        return np.asarray(result.holospectrum, dtype=float).ravel()
    if feature == "am_frequency":
        vectors = [np.asarray(freq, dtype=float).ravel() for freq in result.am_frequency if np.asarray(freq).size]
        if not vectors:
            return np.empty(0)
        return np.concatenate(vectors)
    raise ValueError("feature must be 'mode_energy', 'marginal', 'hht', 'holospectrum', or 'am_frequency'")


def _as_result_list(results: object | list[object]) -> list[object]:
    """Normalize one HHSA result or a list of results into a list."""

    return results if isinstance(results, list) else [results]


def hhsa_feature_matrix(results: object | list[object], feature: str = "mode_energy") -> np.ndarray:
    """Stack HHSA feature vectors into a matrix, padding shorter rows with NaN."""

    vectors = [hhsa_feature(result, feature=feature) for result in _as_result_list(results)]
    if not vectors:
        return np.empty((0, 0))
    width = max((vector.size for vector in vectors), default=0)
    matrix = np.full((len(vectors), width), np.nan)
    for row, vector in enumerate(vectors):
        matrix[row, : vector.size] = vector
    return matrix


def hhsa_t_test(
    group_a: object | list[object],
    group_b: object | list[object],
    *,
    feature: str = "mode_energy",
    equal_var: bool = False,
) -> StatisticalTestResult:
    """Run a feature-wise independent t-test between two HHSA result groups."""

    a = hhsa_feature_matrix(group_a, feature=feature)
    b = hhsa_feature_matrix(group_b, feature=feature)
    width = max(a.shape[1], b.shape[1])
    if a.shape[1] != width:
        a = np.pad(a, ((0, 0), (0, width - a.shape[1])), constant_values=np.nan)
    if b.shape[1] != width:
        b = np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=np.nan)
    statistic, pvalue = stats.ttest_ind(a, b, axis=0, equal_var=equal_var, nan_policy="omit")
    return StatisticalTestResult(
        statistic=np.asarray(statistic, dtype=float),
        pvalue=np.asarray(pvalue, dtype=float),
        feature=feature,
        method="welch_t_test" if not equal_var else "student_t_test",
        n_group_a=a.shape[0],
        n_group_b=b.shape[0],
    )


def hhsa_permutation_test(
    group_a: object | list[object],
    group_b: object | list[object],
    *,
    feature: str = "mode_energy",
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> StatisticalTestResult:
    """Run a two-sided feature-wise permutation test between HHSA result groups."""

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(random_state)
    a = hhsa_feature_matrix(group_a, feature=feature)
    b = hhsa_feature_matrix(group_b, feature=feature)
    width = max(a.shape[1], b.shape[1])
    if a.shape[1] != width:
        a = np.pad(a, ((0, 0), (0, width - a.shape[1])), constant_values=np.nan)
    if b.shape[1] != width:
        b = np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=np.nan)

    combined = np.vstack([a, b])
    n_a = a.shape[0]
    observed = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    extreme = np.zeros(width, dtype=int)
    for _ in range(n_permutations):
        order = rng.permutation(combined.shape[0])
        perm_a = combined[order[:n_a]]
        perm_b = combined[order[n_a:]]
        permuted = np.nanmean(perm_a, axis=0) - np.nanmean(perm_b, axis=0)
        extreme += np.abs(permuted) >= np.abs(observed)
    pvalue = (extreme + 1) / (n_permutations + 1)
    return StatisticalTestResult(
        statistic=observed,
        pvalue=pvalue,
        feature=feature,
        method="permutation_mean_difference",
        n_group_a=a.shape[0],
        n_group_b=b.shape[0],
    )
