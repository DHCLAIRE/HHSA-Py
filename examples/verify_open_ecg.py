"""Verify HHSA on an open ECG signal bundled through SciPy datasets.

SciPy's electrocardiogram example is from the MIT-BIH Arrhythmia Database. If it
is not cached locally, SciPy may try to download it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhsa import marginal_spectrum, mode_energy, normalized_entropy, run_hhsa


def load_ecg_segment(seconds: float = 5.0) -> tuple[np.ndarray, float, str]:
    try:
        from scipy.datasets import electrocardiogram

        sample_rate = 360.0
        data = electrocardiogram()
        n = int(seconds * sample_rate)
        return np.asarray(data[:n], dtype=float), sample_rate, "scipy.datasets.electrocardiogram"
    except Exception:
        sample_rate = 500.0
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        signal = np.sin(2 * np.pi * 9 * t) * (1.0 + 0.35 * np.sin(2 * np.pi * 1.2 * t))
        signal += 0.15 * np.sin(2 * np.pi * 35 * t)
        return signal, sample_rate, "synthetic AM signal fallback"


def main() -> None:
    signal, sample_rate, source = load_ecg_segment()
    signal = signal - np.mean(signal)
    result = run_hhsa(
        signal,
        sample_rate,
        decomposition="iceemdan",
        frequency_method="hybrid",
        max_imfs=4,
        max_am_imfs=3,
        ensemble_size=16,
        noise_width=0.1,
        random_state=42,
    )
    centers, marginal = marginal_spectrum(result.frequency, result.amplitude, bins=64)
    energies = mode_energy(result.imfs)

    print(f"source={source}")
    print(f"samples={signal.size} sample_rate={sample_rate:g}Hz")
    print(f"first_layer_imfs={result.imfs.shape[0]}")
    print(f"mode_energy={np.round(energies, 3).tolist()}")
    print(f"energy_entropy={normalized_entropy(energies):.3f}")
    print(f"dominant_hilbert_frequency={centers[np.argmax(marginal)]:.3f}Hz")


if __name__ == "__main__":
    main()
