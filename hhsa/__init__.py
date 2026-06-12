"""Holo-Hilbert spectral analysis helpers."""

from .decomposition import ICEEMDAN, ceemdan, emd, iceemdan
from .frequency import frequency_transform, generalized_zero_crossing, quadrature_frequency
from .pipeline import HHSAResult, run_hhsa
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
    "ICEEMDAN",
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
    "spectrum_bin_edges",
]
