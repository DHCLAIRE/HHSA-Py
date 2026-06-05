"""Two-layer Holo-Hilbert spectral analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .decomposition import ceemdan, emd, iceemdan
from .frequency import frequency_transform

DecompositionMethod = Literal["emd", "ceemdan", "iceemdan"]
FrequencyMethod = Literal["quad", "gzc", "hybrid"]


@dataclass
class HHSAResult:
    signal: np.ndarray
    sample_rate: float
    imfs: np.ndarray
    residue: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    am_imfs: list[np.ndarray]
    am_residues: list[np.ndarray]
    am_frequency: list[np.ndarray]

    @property
    def reconstruction(self) -> np.ndarray:
        return self.imfs.sum(axis=0) + self.residue

    @property
    def reconstruction_error(self) -> float:
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
    max_imfs: int | None = 6,
    max_am_imfs: int | None = 4,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | None = 13,
    max_siftings: int = 20,
    stop_sd: float = 0.2,
) -> HHSAResult:
    """Run HHSA: layer-1 signal IMFs, then layer-2 amplitude-envelope IMFs."""

    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")

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
        return HHSAResult(x, sample_rate, empty, residue, empty, empty, empty, [], [], [])

    amplitude, phase, frequency = frequency_transform(imfs, sample_rate, method=frequency_method)
    am_imfs: list[np.ndarray] = []
    am_residues: list[np.ndarray] = []
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
            _, _, am_freq = frequency_transform(modes, sample_rate, method=frequency_method)
        else:
            am_freq = np.empty((0, x.size))
        am_frequency.append(am_freq)

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
        am_frequency=am_frequency,
    )
