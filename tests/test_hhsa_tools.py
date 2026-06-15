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
        emd_backend="local",
    )
    results = pipeline.fit(stereo, channel_axis="last")
    summaries = pipeline.summarize(results)

    assert len(results) == 2
    assert len(summaries) == 2
