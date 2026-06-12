"""Class-based HHSA interface for homework and notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hhsa import HHSAResult, ICEEMDAN, mode_energy, run_hhsa


@dataclass
class HHSAAnalyzer:
    """Analyze one-dimensional non-stationary neural signals with HHSA."""

    sample_rate: float
    decomposition: str = "iceemdan"
    frequency_method: str = "hybrid"
    max_imfs: int | None = 6
    max_am_imfs: int | None = 4
    ensemble_size: int = 64
    noise_width: float = 0.2
    random_state: int | None = 13

    def fit(self, signal: np.ndarray) -> HHSAResult:
        """Run the two-layer HHSA pipeline and return the full result."""

        return run_hhsa(
            signal,
            self.sample_rate,
            decomposition=self.decomposition,
            frequency_method=self.frequency_method,
            max_imfs=self.max_imfs,
            max_am_imfs=self.max_am_imfs,
            ensemble_size=self.ensemble_size,
            noise_width=self.noise_width,
            random_state=self.random_state,
        )

    def summarize(self, result: HHSAResult, *, bins: int = 128) -> dict[str, np.ndarray | float]:
        """Return common summary statistics for an HHSA result."""

        energies = mode_energy(result.imfs)
        return {
            "mode_energy": energies,
            "frequency_bins": result.carrier_bins,
            "marginal_spectrum": result.marginal,
            "hht": result.hht,
            "am_frequency_bins": result.am_bins,
            "holospectrum": result.holospectrum,
            "reconstruction_error": result.reconstruction_error,
        }
