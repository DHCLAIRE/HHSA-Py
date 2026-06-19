"""Holo-Hilbert spectral analysis helpers."""

from .decomposition import ceemdan, emd, ensemble_sift, iceemdan, mask_sift
from .frequency import frequency_transform, generalized_zero_crossing, quadrature_frequency
from .pipeline import HHSAResult, as_channel_matrix, run_hhsa, run_hhsa_dataset
from .statistics import (
    StatisticalTestResult,
    hhsa_feature,
    hhsa_feature_matrix,
    hhsa_permutation_test,
    hhsa_t_test,
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
    "StatisticalTestResult",
    "as_channel_matrix",
    "ceemdan",
    "emd",
    "ensemble_sift",
    "frequency_transform",
    "generalized_zero_crossing",
    "hhsa_feature",
    "hhsa_feature_matrix",
    "hhsa_permutation_test",
    "hhsa_t_test",
    "hilbert_huang_spectrum",
    "holospectrum",
    "iceemdan",
    "marginal_spectrum",
    "mask_sift",
    "mode_energy",
    "normalized_entropy",
    "orthogonality_index",
    "quadrature_frequency",
    "run_hhsa",
    "run_hhsa_dataset",
    "spectrum_bin_edges",
]
