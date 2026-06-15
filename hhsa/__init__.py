"""Holo-Hilbert spectral analysis helpers."""

from .decomposition import ceemdan, emd, iceemdan
from .frequency import frequency_transform, generalized_zero_crossing, quadrature_frequency
from .pipeline import HHSAResult, as_channel_matrix, run_hhsa, run_hhsa_dataset
from .statistics import (
    hilbert_huang_spectrum,
    holospectrum,
    marginal_spectrum,
    mode_energy,
    normalized_entropy,
    orthogonality_index,
    spectrum_bin_edges,
)

__all__ = [
    "HHSAResult",
    "as_channel_matrix",
    "ceemdan",
    "emd",
    "frequency_transform",
    "generalized_zero_crossing",
    "hilbert_huang_spectrum",
    "holospectrum",
    "iceemdan",
    "marginal_spectrum",
    "mode_energy",
    "normalized_entropy",
    "orthogonality_index",
    "quadrature_frequency",
    "run_hhsa",
    "run_hhsa_dataset",
    "spectrum_bin_edges",
]
