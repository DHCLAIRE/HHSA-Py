import numpy as np

from hhsa import generalized_zero_crossing, mode_energy, quadrature_frequency, run_hhsa


def test_frequency_estimators_track_sine():
    sample_rate = 200.0
    t = np.arange(0, 1, 1 / sample_rate)
    x = np.sin(2 * np.pi * 12 * t)

    _, _, quad = quadrature_frequency(x, sample_rate)
    gzc = generalized_zero_crossing(x, sample_rate)

    assert np.median(quad[20:-20]) == np.median(quad[20:-20])
    assert abs(np.median(quad[20:-20]) - 12) < 0.5
    assert abs(np.median(gzc[20:-20]) - 12) < 1.0


def test_hhsa_pipeline_returns_modes_for_am_signal():
    sample_rate = 200.0
    t = np.arange(0, 1, 1 / sample_rate)
    x = (1 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 18 * t)

    result = run_hhsa(
        x,
        sample_rate,
        decomposition="iceemdan",
        frequency_method="hybrid",
        max_imfs=3,
        max_am_imfs=2,
        ensemble_size=8,
        noise_width=0.05,
        random_state=1,
        max_siftings=8,
    )

    assert result.imfs.shape[1] == x.size
    assert result.frequency.shape == result.imfs.shape
    assert len(result.am_imfs) == result.imfs.shape[0]
    assert np.all(mode_energy(result.imfs) > 0)
