import matplotlib
import numpy as np

matplotlib.use("Agg")

from hhsa import plot_am_fm, plot_decomposition, plot_sifting_options, run_hhsa


def test_plot_decomposition_returns_figure():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 8 * t)
    imfs = signal[np.newaxis, :]
    residue = np.zeros_like(signal)

    fig, axes = plot_decomposition(signal, imfs, residue, sample_rate=sample_rate)

    assert fig is not None
    assert len(axes) == 3


def test_plot_sifting_options_returns_figure():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 8 * t)

    fig, axes = plot_sifting_options(
        signal,
        sample_rate,
        methods=("sift", "ensemble_sift", "mask_sift"),
        max_imfs=2,
        ensemble_size=4,
        mask_freqs=8 / sample_rate,
    )

    assert fig is not None
    assert len(axes) == 3


def test_plot_am_fm_returns_figure():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 8 * t)
    result = run_hhsa(
        signal,
        sample_rate,
        decomposition="sift",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
    )

    fig, axes = plot_am_fm(result)

    assert fig is not None
    assert axes.shape[1] == 2
