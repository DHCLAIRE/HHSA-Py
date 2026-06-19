"""Two-layer Holo-Hilbert spectral analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import wavfile

from .decomposition import EMDBackend, ceemdan, emd, ensemble_sift, iceemdan, mask_sift
from .frequency import frequency_transform
from .statistics import SpectrumBins, hilbert_huang_spectrum, holospectrum, spectrum_bin_edges

# Supported first- and second-layer decomposition method names.
DecompositionMethod = Literal["emd", "sift", "ensemble_sift", "mask_sift", "ceemdan", "iceemdan"]

# Supported instantaneous-frequency estimator names.
FrequencyMethod = Literal["quad", "gzc", "hybrid"]

# Supported array orientation hints for multi-channel arrays.
ChannelAxis = Literal["auto", "first", "last"]


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
    emd_backend: EMDBackend,
    mask_freqs: np.ndarray | float | None,
    mask_amp: float,
    mask_amp_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested decomposition function with shared settings."""

    if method in {"emd", "sift"}:
        return emd(
            signal,
            max_imfs=max_imfs,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            backend=emd_backend,
            sift_method="sift",
        )
    if method == "ensemble_sift":
        return ensemble_sift(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    if method == "mask_sift":
        return mask_sift(
            signal,
            max_imfs=max_imfs,
            mask_freqs=mask_freqs,
            mask_amp=mask_amp,
            mask_amp_mode=mask_amp_mode,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    if method == "ceemdan":
        return ceemdan(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            emd_backend=emd_backend,
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
            emd_backend=emd_backend,
        )
    raise ValueError("method must be 'emd', 'sift', 'ensemble_sift', 'mask_sift', 'ceemdan', or 'iceemdan'")


def _scale_audio_samples(data: np.ndarray) -> np.ndarray:
    """Convert integer audio samples to floating point values near [-1, 1]."""

    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        return arr.astype(float) / max_value
    return arr.astype(float)


def _read_wav(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Read mono or multi-channel WAV audio as channels x samples."""

    sample_rate, data = wavfile.read(path)
    arr = _scale_audio_samples(data)
    if arr.ndim == 1:
        return arr[np.newaxis, :], float(sample_rate), ["audio_0"]
    return arr.T, float(sample_rate), [f"audio_{idx}" for idx in range(arr.shape[1])]


def _read_mne_data(data: object, picks: object | None) -> tuple[np.ndarray, float, list[str]]:
    """Read Raw, Epochs, or Evoked-like MNE objects as channels x samples."""

    info = getattr(data, "info", {})
    sample_rate = float(info["sfreq"])
    try:
        values = data.get_data(picks=picks)
    except TypeError:
        values = data.get_data()
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        n_epochs, n_channels, n_times = arr.shape
        arr = arr.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
    if arr.ndim != 2:
        raise ValueError("MNE data must resolve to shape (channels, samples)")
    names = list(info.get("ch_names", [f"channel_{idx}" for idx in range(arr.shape[0])]))
    if len(names) != arr.shape[0]:
        names = [f"channel_{idx}" for idx in range(arr.shape[0])]
    return arr, sample_rate, names


def as_channel_matrix(
    data: object,
    *,
    sample_rate: float | None = None,
    channel_axis: ChannelAxis = "auto",
    picks: object | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Normalize EEG, MEG, audio, or array input to channels x samples.

    Accepts 1-D arrays, 2-D arrays, WAV paths, and MNE Raw/Epochs/Evoked-like
    objects. For 2-D arrays, ``channel_axis="auto"`` treats the smaller
    dimension as channels.
    """

    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() != ".wav":
            raise ValueError("only .wav audio paths are supported")
        return _read_wav(path)

    if hasattr(data, "get_data") and hasattr(data, "info"):
        return _read_mne_data(data, picks)

    if sample_rate is None:
        raise ValueError("sample_rate is required for NumPy array input")
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr[np.newaxis, :], float(sample_rate), ["signal_0"]
    if arr.ndim != 2:
        raise ValueError("array input must have shape (samples,), (channels, samples), or (samples, channels)")
    if channel_axis == "first":
        matrix = arr
    elif channel_axis == "last":
        matrix = arr.T
    elif channel_axis == "auto":
        matrix = arr if arr.shape[0] <= arr.shape[1] else arr.T
    else:
        raise ValueError("channel_axis must be 'auto', 'first', or 'last'")
    return matrix, float(sample_rate), [f"channel_{idx}" for idx in range(matrix.shape[0])]


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
    emd_backend: EMDBackend = "auto",
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
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
        emd_backend=emd_backend,
        mask_freqs=mask_freqs,
        mask_amp=mask_amp,
        mask_amp_mode=mask_amp_mode,
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
            emd_backend=emd_backend,
            mask_freqs=mask_freqs,
            mask_amp=mask_amp,
            mask_amp_mode=mask_amp_mode,
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


def run_hhsa_dataset(
    data: object,
    sample_rate: float | None = None,
    *,
    channel_axis: ChannelAxis = "auto",
    picks: object | None = None,
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
    emd_backend: EMDBackend = "auto",
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
) -> list[HHSAResult]:
    """Run HHSA independently for every channel in EEG, MEG, or audio data."""

    matrix, inferred_rate, _ = as_channel_matrix(data, sample_rate=sample_rate, channel_axis=channel_axis, picks=picks)
    results: list[HHSAResult] = []
    for channel_index, channel in enumerate(matrix):
        seed = None if random_state is None else random_state + channel_index
        results.append(
            run_hhsa(
                channel,
                inferred_rate,
                decomposition=decomposition,
                frequency_method=frequency_method,
                max_imfs=max_imfs,
                max_am_imfs=max_am_imfs,
                ensemble_size=ensemble_size,
                noise_width=noise_width,
                random_state=seed,
                max_siftings=max_siftings,
                stop_sd=stop_sd,
                carrier_hist=carrier_hist,
                am_hist=am_hist,
                emd_backend=emd_backend,
                mask_freqs=mask_freqs,
                mask_amp=mask_amp,
                mask_amp_mode=mask_amp_mode,
            )
        )
    return results
