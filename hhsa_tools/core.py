"""Class-based HHSA pipeline interface for homework and notebooks.

This module intentionally exposes one high-level tool: :class:`HHSAPipeline`.
The pipeline stores repeated HHSA settings and delegates the numerical work to
the lower-level functions in :mod:`hhsa`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hhsa import HHSAResult, mode_energy, run_hhsa


@dataclass
class HHSAPipeline:
    """Run HHSA on one-dimensional non-stationary signals.

    Parameters
    ----------
    sample_rate:
        Sampling rate of the input signal in Hz.
    decomposition:
        First- and second-layer decomposition method. Supported values are
        ``"iceemdan"``, ``"ceemdan"``, and ``"emd"``.
    frequency_method:
        Instantaneous-frequency estimator. Use ``"quad"`` for Hilbert phase,
        ``"gzc"`` for Generalized Zero-Crossing, or ``"hybrid"`` to combine
        both.
    max_imfs:
        Maximum number of first-layer carrier IMFs to extract. Defaults to 10.
    max_am_imfs:
        Maximum number of second-layer amplitude-modulation IMFs to extract
        from each carrier amplitude envelope.
    ensemble_size:
        Number of noise realizations for CEEMDAN/ICEEMDAN decompositions.
    noise_width:
        Relative noise scale for ensemble decompositions.
    random_state:
        Optional seed for reproducible ensemble noise.
    """

    sample_rate: float
    decomposition: str = "iceemdan"
    frequency_method: str = "hybrid"
    max_imfs: int | None = 10
    max_am_imfs: int | None = 4
    ensemble_size: int = 64
    noise_width: float = 0.2
    random_state: int | None = 13

    def fit(self, signal: np.ndarray) -> HHSAResult:
        """Run the two-layer HHSA pipeline with the stored settings.

        Parameters
        ----------
        signal:
            One-dimensional time series to analyze.

        Returns
        -------
        HHSAResult
            Full pipeline output, including first-layer IMFs, second-layer
            amplitude-envelope IMFs, instantaneous frequency/amplitude arrays,
            HHT, holospectrum, and reconstruction diagnostics.
        """

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
        """Collect common summary arrays from an HHSA result.

        Parameters
        ----------
        result:
            Output returned by :meth:`fit` or by :func:`hhsa.run_hhsa`.
        bins:
            Kept for backwards compatibility with older summaries. The current
            summary uses the frequency bins already stored on ``result``.

        Returns
        -------
        dict
            Dictionary with mode energy, carrier and AM frequency bins,
            marginal spectrum, HHT, holospectrum, and reconstruction error.
        """

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
