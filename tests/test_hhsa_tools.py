import numpy as np

from hhsa_tools import HHSAAnalyzer


def test_hhsa_tools_public_analyzer_imports_and_runs():
    sample_rate = 100.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 12 * t)

    analyzer = HHSAAnalyzer(
        sample_rate=sample_rate,
        decomposition="emd",
        frequency_method="quad",
        max_imfs=2,
        max_am_imfs=1,
    )
    result = analyzer.fit(signal)
    summary = analyzer.summarize(result)

    assert result.imfs.shape[1] == signal.size
    assert "mode_energy" in summary
