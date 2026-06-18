import numpy as np

from hhsa import ceemdan, generalized_zero_crossing, iceemdan, mode_energy, quadrature_frequency, run_hhsa, run_hhsa_dataset


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
    assert len(result.am_amplitude) == result.imfs.shape[0]
    assert result.hht.shape == (result.carrier_bins.size, x.size)
    assert result.holospectrum.shape == (result.carrier_bins.size, result.am_bins.size)
    assert np.all(mode_energy(result.imfs) > 0)


def test_hhsa_dataset_runs_multichannel_array():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 14 * t),
        ]
    )

    results = run_hhsa_dataset(
        data,
        sample_rate,
        decomposition="emd",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
        emd_backend="emd-python",
    )

    assert len(results) == 2
    assert all(result.imfs.shape[1] == t.size for result in results)


def test_iceemdan_function_reconstructs_signal():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    x = np.sin(2 * np.pi * 12 * t) + 0.4 * np.sin(2 * np.pi * 4 * t)

    imfs, residue = iceemdan(
        x,
        max_imfs=3,
        ensemble_size=6,
        noise_width=0.05,
        random_state=7,
        max_siftings=8,
    )

    assert imfs.shape[1] == x.size
    np.testing.assert_allclose(imfs.sum(axis=0) + residue, x, atol=1e-10)


def test_ceemdan_uses_external_pyemd_shape():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    x = np.sin(2 * np.pi * 8 * t)

    imfs, residue = ceemdan(x, max_imfs=2, ensemble_size=4, random_state=2)

    assert imfs.shape[1] == x.size
    np.testing.assert_allclose(imfs.sum(axis=0) + residue, x, atol=1e-10)
