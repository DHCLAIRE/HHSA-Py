"""Class-based HHSA pipeline interface for homework and notebooks.

This module intentionally exposes one high-level tool: :class:`HHSAPipeline`.
The pipeline stores repeated HHSA settings and delegates the numerical work to
the lower-level functions in :mod:`hhsa`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hhsa import HHSAResult, as_channel_matrix, mode_energy, run_hhsa, run_hhsa_dataset
from hhsa.decomposition import EMDBackend
from hhsa.pipeline import ChannelAxis


@dataclass
class HHSAPipeline:
    """Run HHSA on one-dimensional non-stationary signals.

    Parameters
    ----------
    sample_rate:
        Sampling rate of array input in Hz. MNE objects and WAV paths can
        provide this value automatically.
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
    emd_backend:
        EMD implementation to use. ``"auto"`` tries EMD-Python, then PyEMD,
        using only imported EMD libraries.
    """

    sample_rate: float | None = None
    decomposition: str = "iceemdan"
    frequency_method: str = "hybrid"
    max_imfs: int | None = 10
    max_am_imfs: int | None = 4
    ensemble_size: int = 64
    noise_width: float = 0.2
    random_state: int | None = 13
    emd_backend: EMDBackend = "auto"

    def fit(
        self,
        signal: object,
        *,
        channel_axis: ChannelAxis = "auto",
        picks: object | None = None,
    ) -> HHSAResult | list[HHSAResult]:
        """Run the two-layer HHSA pipeline with the stored settings.

        Parameters
        ----------
        signal:
            One-dimensional signal, 2-D channel array, WAV path, or MNE
            Raw/Epochs/Evoked-like object.
        channel_axis:
            Orientation hint for 2-D array input. Use ``"first"`` for
            channels x samples, ``"last"`` for samples x channels, or
            ``"auto"`` to infer the smaller dimension as channels.
        picks:
            Optional MNE channel picks passed to ``get_data``.

        Returns
        -------
        HHSAResult or list[HHSAResult]
            A single result for one-channel input, or one result per channel
            for EEG, MEG, or multi-channel audio data.
        """

        matrix, sample_rate, _ = as_channel_matrix(
            signal,
            sample_rate=self.sample_rate,
            channel_axis=channel_axis,
            picks=picks,
        )
        if matrix.shape[0] > 1:
            return run_hhsa_dataset(
                signal,
                sample_rate=self.sample_rate,
                channel_axis=channel_axis,
                picks=picks,
                decomposition=self.decomposition,
                frequency_method=self.frequency_method,
                max_imfs=self.max_imfs,
                max_am_imfs=self.max_am_imfs,
                ensemble_size=self.ensemble_size,
                noise_width=self.noise_width,
                random_state=self.random_state,
                emd_backend=self.emd_backend,
            )
        return run_hhsa(
            matrix[0],
            sample_rate,
            decomposition=self.decomposition,
            frequency_method=self.frequency_method,
            max_imfs=self.max_imfs,
            max_am_imfs=self.max_am_imfs,
            ensemble_size=self.ensemble_size,
            noise_width=self.noise_width,
            random_state=self.random_state,
            emd_backend=self.emd_backend,
        )

    def summarize(
        self,
        result: HHSAResult | list[HHSAResult],
        *,
        bins: int = 128,
    ) -> dict[str, np.ndarray | float] | list[dict[str, np.ndarray | float]]:
        """Collect common summary arrays from an HHSA result.

        Parameters
        ----------
        result:
            Output returned by :meth:`fit`, :func:`hhsa.run_hhsa`, or
            :func:`hhsa.run_hhsa_dataset`.
        bins:
            Kept for backwards compatibility with older summaries. The current
            summary uses the frequency bins already stored on ``result``.

        Returns
        -------
        dict
            Dictionary for one channel, or a list of dictionaries for
            multi-channel data.
        """

        if isinstance(result, list):
            return [self.summarize(channel_result, bins=bins) for channel_result in result]
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
