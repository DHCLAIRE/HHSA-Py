"""Two-layer Holo-Hilbert spectral analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .decomposition import ceemdan, emd, iceemdan
from .frequency import frequency_transform
from .statistics import SpectrumBins, hilbert_huang_spectrum, holospectrum, spectrum_bin_edges

# Supported first- and second-layer decomposition method names.
DecompositionMethod = Literal["emd", "ceemdan", "iceemdan"]

# Supported instantaneous-frequency estimator names.
FrequencyMethod = Literal["quad", "gzc", "hybrid"]


@dataclass
class HHSAResult:
    """Container for all arrays produced by the two-layer HHSA pipeline."""

    signal: np.ndarray
    sample_rate: float
    imfs: np.ndarray
    residue: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    am_imfs: list[np.ndarray]
    am_residues: list[np.ndarray]
    am_amplitude: list[np.ndarray]
    am_phase: list[np.ndarray]
    am_frequency: list[np.ndarray]
    carrier_bins: np.ndarray
    am_bins: np.ndarray
    marginal: np.ndarray
    hht: np.ndarray
    holospectrum: np.ndarray

    @property
    def reconstruction(self) -> np.ndarray:
        """Reconstruct the signal from first-layer IMFs and final residue."""

        return self.imfs.sum(axis=0) + self.residue

    @property
    def reconstruction_error(self) -> float:
        """Return relative reconstruction error against the original signal."""

        denom = np.linalg.norm(self.signal) + np.finfo(float).eps
        return float(np.linalg.norm(self.signal - self.reconstruction) / denom)


def _decompose(
    signal: np.ndarray,
    method: DecompositionMethod,
    *,
    max_imfs: int | None,
    ensemble_size: int,
    noise_width: float,
    random_state: int | None,
    max_siftings: int,
    stop_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested decomposition function with shared settings."""

    if method == "emd":
        return emd(signal, max_imfs=max_imfs, max_siftings=max_siftings, stop_sd=stop_sd)
    if method == "ceemdan":
        return ceemdan(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    if method == "iceemdan":
        return iceemdan(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    raise ValueError("method must be 'emd', 'ceemdan', or 'iceemdan'")


def run_hhsa(
    signal: np.ndarray,
    sample_rate: float,
    *,
    decomposition: DecompositionMethod = "iceemdan",
    frequency_method: FrequencyMethod = "hybrid",
    max_imfs: int | None = 10,
    max_am_imfs: int | None = 4,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | None = 13,
    max_siftings: int = 20,
    stop_sd: float = 0.2,
    carrier_hist: SpectrumBins | None = None,
    am_hist: SpectrumBins | None = None,
) -> HHSAResult:
    """Run HHSA using the Holo-Hilbert spectrum pipeline.

    The stages mirror the holospectrum tutorial:
    first-layer decomposition, instantaneous phase/frequency/amplitude,
    second-layer decomposition of each instantaneous-amplitude trace, then
    Hilbert-Huang and Holo-Hilbert spectral histograms.
    """

    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if carrier_hist is None:
        carrier_hist = (max(sample_rate / x.size, 1e-6), sample_rate / 2.0, 128, "log")
    if am_hist is None:
        am_hist = (max(sample_rate / x.size, 1e-6), sample_rate / 4.0, 64, "log")

    imfs, residue = _decompose(
        x,
        decomposition,
        max_imfs=max_imfs,
        ensemble_size=ensemble_size,
        noise_width=noise_width,
        random_state=random_state,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
    )
    if imfs.size == 0:
        empty = np.empty((0, x.size))
        carrier_bins, marginal, hht = hilbert_huang_spectrum(empty, empty, carrier_hist)
        am_bins, _ = spectrum_bin_edges(am_hist)
        return HHSAResult(
            x,
            sample_rate,
            empty,
            residue,
            empty,
            empty,
            empty,
            [],
            [],
            [],
            [],
            [],
            carrier_bins,
            am_bins,
            marginal,
            hht,
            np.zeros((carrier_bins.size, am_bins.size)),
        )

    amplitude, phase, frequency = frequency_transform(imfs, sample_rate, method=frequency_method)
    am_imfs: list[np.ndarray] = []
    am_residues: list[np.ndarray] = []
    am_amplitude: list[np.ndarray] = []
    am_phase: list[np.ndarray] = []
    am_frequency: list[np.ndarray] = []

    for mode_index, envelope in enumerate(amplitude):
        seed = None if random_state is None else random_state + mode_index + 1
        modes, am_residue = _decompose(
            envelope - np.mean(envelope),
            decomposition,
            max_imfs=max_am_imfs,
            ensemble_size=max(8, ensemble_size // 2),
            noise_width=noise_width,
            random_state=seed,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
        am_imfs.append(modes)
        am_residues.append(am_residue)
        if modes.size:
            am_amp, am_ip, am_freq = frequency_transform(modes, sample_rate, method=frequency_method)
        else:
            am_amp = np.empty((0, x.size))
            am_ip = np.empty((0, x.size))
            am_freq = np.empty((0, x.size))
        am_amplitude.append(am_amp)
        am_phase.append(am_ip)
        am_frequency.append(am_freq)

    carrier_bins, marginal, hht = hilbert_huang_spectrum(frequency, amplitude, carrier_hist)
    carrier_bins, am_bins, holo = holospectrum(frequency, am_frequency, am_amplitude, carrier_hist, am_hist)

    return HHSAResult(
        signal=x,
        sample_rate=sample_rate,
        imfs=imfs,
        residue=residue,
        amplitude=amplitude,
        phase=phase,
        frequency=frequency,
        am_imfs=am_imfs,
        am_residues=am_residues,
        am_amplitude=am_amplitude,
        am_phase=am_phase,
        am_frequency=am_frequency,
        carrier_bins=carrier_bins,
        am_bins=am_bins,
        marginal=marginal,
        hht=hht,
        holospectrum=holo,
    )
