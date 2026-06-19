"""Visualization helpers for decomposition, AM, and FM outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .decomposition import DecompositionMethod, EMDBackend, decompose_signal
from .pipeline import HHSAResult


def _get_pyplot():
    """Import matplotlib lazily so plotting remains an optional dependency."""

    import matplotlib.pyplot as plt

    return plt


def _time_axis(n_samples: int, sample_rate: float | None) -> np.ndarray:
    """Return sample indices or seconds for plot x-axes."""

    if sample_rate is None:
        return np.arange(n_samples)
    return np.arange(n_samples) / sample_rate


def plot_decomposition(
    signal: np.ndarray,
    imfs: np.ndarray,
    residue: np.ndarray,
    *,
    sample_rate: float | None = None,
    title: str = "Signal decomposition",
):
    """Plot a signal, its IMFs, and final residue."""

    plt = _get_pyplot()
    x = np.asarray(signal, dtype=float)
    modes = np.asarray(imfs, dtype=float)
    residual = np.asarray(residue, dtype=float)
    time = _time_axis(x.size, sample_rate)
    n_rows = modes.shape[0] + 2
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, max(4, 1.4 * n_rows)))
    axes[0].plot(time, x, color="black", linewidth=1)
    axes[0].set_ylabel("Signal")
    axes[0].set_title(title)
    for idx, mode in enumerate(modes, start=1):
        axes[idx].plot(time, mode, linewidth=1)
        axes[idx].set_ylabel(f"IMF {idx}")
    axes[-1].plot(time, residual, color="tab:red", linewidth=1)
    axes[-1].set_ylabel("Residue")
    axes[-1].set_xlabel("Time (s)" if sample_rate is not None else "Sample")
    fig.tight_layout()
    return fig, axes


def plot_sifting_options(
    signal: np.ndarray,
    sample_rate: float,
    *,
    methods: Sequence[DecompositionMethod] = (
        "sift",
        "ensemble_sift",
        "complete_ensemble_sift",
        "mask_sift",
        "iterated_mask_sift",
        "ceemdan",
        "iceemdan",
    ),
    max_imfs: int | None = 5,
    ensemble_size: int = 32,
    noise_width: float = 0.2,
    random_state: int | None = 13,
    max_siftings: int = 20,
    stop_sd: float = 0.2,
    emd_backend: EMDBackend = "auto",
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
):
    """Plot IMF stacks from all requested EMD/CEEMDAN/ICEEMDAN options."""

    plt = _get_pyplot()
    x = np.asarray(signal, dtype=float)
    time = _time_axis(x.size, sample_rate)
    fig, axes = plt.subplots(len(methods), 1, sharex=True, figsize=(11, max(4, 2.2 * len(methods))))
    if len(methods) == 1:
        axes = np.asarray([axes])
    for axis, method in zip(axes, methods):
        imfs, residue = decompose_signal(
            x,
            method,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            emd_backend=emd_backend,
            mask_freqs=mask_freqs,
            mask_amp=mask_amp,
            mask_amp_mode=mask_amp_mode,
        )
        offset_scale = np.nanmax(np.abs(x)) or 1.0
        for idx, mode in enumerate(imfs):
            axis.plot(time, mode + idx * offset_scale * 1.2, linewidth=0.9)
        axis.plot(time, residue + imfs.shape[0] * offset_scale * 1.2, color="tab:red", linewidth=0.9)
        axis.set_ylabel(str(method))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sifting/decomposition option comparison")
    fig.tight_layout()
    return fig, axes


def plot_am_fm(
    result: HHSAResult,
    *,
    max_modes: int | None = None,
):
    """Plot instantaneous amplitude and frequency tracks for HHSA IMFs."""

    plt = _get_pyplot()
    n_modes = result.imfs.shape[0] if max_modes is None else min(max_modes, result.imfs.shape[0])
    time = _time_axis(result.signal.size, result.sample_rate)
    fig, axes = plt.subplots(n_modes, 2, sharex=True, figsize=(12, max(4, 2.0 * n_modes)))
    if n_modes == 1:
        axes = np.asarray([axes])
    for idx in range(n_modes):
        axes[idx, 0].plot(time, result.amplitude[idx], color="tab:blue", linewidth=1)
        axes[idx, 0].set_ylabel(f"IMF {idx + 1} AM")
        axes[idx, 1].plot(time, result.frequency[idx], color="tab:orange", linewidth=1)
        axes[idx, 1].set_ylabel(f"IMF {idx + 1} FM")
    axes[0, 0].set_title("Instantaneous amplitude")
    axes[0, 1].set_title("Instantaneous frequency")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, axes
