import numpy as np

from hhsa_tools import HHSAPipeline


def test_hhsa_tools_public_pipeline_imports_and_runs():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 12 * t)

    pipeline = HHSAPipeline(
        sample_rate=sample_rate,
        decomposition="emd",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
    )
    result = pipeline.fit(signal)
    summary = pipeline.summarize(result)

    assert result.imfs.shape[1] == signal.size
    assert "mode_energy" in summary


def test_hhsa_pipeline_fit_accepts_audio_style_array():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    left = np.sin(2 * np.pi * 8 * t)
    right = np.sin(2 * np.pi * 12 * t)
    stereo = np.column_stack([left, right])

    pipeline = HHSAPipeline(
        sample_rate=sample_rate,
        decomposition="emd",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
        emd_backend="emd-python",
    )
    results = pipeline.fit(stereo, channel_axis="last")
    summaries = pipeline.summarize(results)

    assert len(results) == 2
    assert len(summaries) == 2


def test_hhsa_pipeline_statistics_methods():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    group_a = np.vstack([np.sin(2 * np.pi * 8 * t), 1.1 * np.sin(2 * np.pi * 8 * t)])
    group_b = np.vstack([1.4 * np.sin(2 * np.pi * 8 * t), 1.5 * np.sin(2 * np.pi * 8 * t)])
    pipeline = HHSAPipeline(
        sample_rate=sample_rate,
        decomposition="sift",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
    )

    results_a = pipeline.fit(group_a, channel_axis="first")
    results_b = pipeline.fit(group_b, channel_axis="first")
    t_result = pipeline.t_test(results_a, results_b)
    permutation = pipeline.permutation_test(results_a, results_b, n_permutations=20, random_state=2)

    assert t_result.method == "welch_t_test"
    assert permutation.method == "permutation_mean_difference"
