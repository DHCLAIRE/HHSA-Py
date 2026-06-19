"""Holo-Hilbert spectral analysis helpers."""

from .decomposition import (
    SiftAcceleration,
    ceemdan,
    complete_ensemble_sift,
    decompose_signal,
    emd,
    ensemble_sift,
    iceemdan,
    iterated_mask_sift,
    mask_sift,
)
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
from .visualization import plot_am_fm, plot_decomposition, plot_sifting_options

__all__ = [
    "HHSAResult",
    "SiftAcceleration",
    "StatisticalTestResult",
    "as_channel_matrix",
    "ceemdan",
    "complete_ensemble_sift",
    "decompose_signal",
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
    "iterated_mask_sift",
    "marginal_spectrum",
    "mask_sift",
    "mode_energy",
    "normalized_entropy",
    "orthogonality_index",
    "plot_am_fm",
    "plot_decomposition",
    "plot_sifting_options",
    "quadrature_frequency",
    "run_hhsa",
    "run_hhsa_dataset",
    "spectrum_bin_edges",
]
