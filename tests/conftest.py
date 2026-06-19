import sys
import types

import numpy as np


def _fake_imfs(signal: np.ndarray, max_imfs: int | None = None) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    mode = x - np.mean(x)
    if np.allclose(mode, 0):
        return np.empty((0, x.size))
    if max_imfs == 0:
        return np.empty((0, x.size))
    return mode[np.newaxis, :]


def pytest_configure():
    emd_module = types.ModuleType("emd")
    sift_module = types.ModuleType("emd.sift")

    def sift(signal, max_imfs=None, imf_opts=None):
        modes = _fake_imfs(signal, max_imfs=max_imfs)
        return modes.T

    def ensemble_sift(signal, max_imfs=None, nensembles=4, ensemble_noise=0.2, noise_seed=None, imf_opts=None):
        modes = _fake_imfs(signal, max_imfs=max_imfs)
        return modes.T

    def complete_ensemble_sift(
        signal,
        max_imfs=None,
        nensembles=4,
        ensemble_noise=0.2,
        noise_seed=None,
        imf_opts=None,
    ):
        modes = _fake_imfs(signal, max_imfs=max_imfs)
        return modes.T

    def mask_sift(
        signal,
        max_imfs=None,
        mask_freqs=None,
        mask_amp=1.0,
        mask_amp_mode="ratio_sig",
        imf_opts=None,
    ):
        modes = _fake_imfs(signal, max_imfs=max_imfs)
        return modes.T

    def iterated_mask_sift(
        signal,
        max_imfs=None,
        mask_0=None,
        mask_amp=1.0,
        mask_amp_mode="ratio_sig",
        imf_opts=None,
    ):
        modes = _fake_imfs(signal, max_imfs=max_imfs)
        return modes.T

    spectra_module = types.ModuleType("emd.spectra")

    def frequency_transform(imf, sample_rate, method, smooth_freq=3, smooth_phase=5):
        values = np.asarray(imf, dtype=float)
        amplitude = np.abs(values)
        phase = np.unwrap(np.angle(values + 1j * np.gradient(values, axis=0)), axis=0)
        frequency = np.maximum(np.gradient(phase, axis=0) * sample_rate / (2 * np.pi), 0)
        return phase, frequency, amplitude

    def hilberthuang(IF, IA, edges=None, sum_time=True, sum_imfs=True, mode="power", sample_rate=1, **kwargs):
        freq = np.asarray(IF, dtype=float)
        amp = np.asarray(IA, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hht = np.zeros((centers.size, freq.shape[0]))
        for sample_index in range(freq.shape[0]):
            for mode_index in range(freq.shape[1]):
                bin_index = np.searchsorted(edges, freq[sample_index, mode_index], side="right") - 1
                if 0 <= bin_index < centers.size:
                    hht[bin_index, sample_index] += amp[sample_index, mode_index] ** 2
        if sum_time:
            hht = hht.sum(axis=1)
        return centers, hht

    def holospectrum(
        IF,
        IF2,
        IA2,
        edges=None,
        edges2=None,
        sum_time=True,
        sum_first_imfs=True,
        sum_second_imfs=True,
        mode="power",
        sample_rate=1,
        **kwargs,
    ):
        carrier_centers = 0.5 * (edges[:-1] + edges[1:])
        am_centers = 0.5 * (edges2[:-1] + edges2[1:])
        return carrier_centers, am_centers, np.zeros((carrier_centers.size, am_centers.size))

    sift_module.sift = sift
    sift_module.ensemble_sift = ensemble_sift
    sift_module.complete_ensemble_sift = complete_ensemble_sift
    sift_module.mask_sift = mask_sift
    sift_module.iterated_mask_sift = iterated_mask_sift
    spectra_module.frequency_transform = frequency_transform
    spectra_module.hilberthuang = hilberthuang
    spectra_module.holospectrum = holospectrum
    emd_module.sift = sift_module
    emd_module.spectra = spectra_module
    sys.modules.setdefault("emd", emd_module)
    sys.modules.setdefault("emd.sift", sift_module)
    sys.modules.setdefault("emd.spectra", spectra_module)

    pyemd_module = types.ModuleType("PyEMD")

    class EMD:
        MAX_ITERATION = 50
        FIXE_H = 50
        std_thr = 0.2
        range_thr = 0.1

        def emd(self, signal, T=None, max_imf=-1):
            limit = None if max_imf == -1 else max_imf
            return _fake_imfs(signal, max_imfs=limit)

    class CEEMDAN:
        def __init__(self, trials=100, epsilon=0.005, ext_EMD=None, parallel=False, seed=None, **kwargs):
            self.ext_EMD = ext_EMD or EMD()

        def ceemdan(self, signal, T=None, max_imf=-1, progress=False):
            limit = None if max_imf == -1 else max_imf
            imfs = _fake_imfs(signal, max_imfs=limit)
            residue = np.asarray(signal, dtype=float) - imfs.sum(axis=0)
            return np.vstack([imfs, residue[np.newaxis, :]])

    pyemd_module.EMD = EMD
    pyemd_module.CEEMDAN = CEEMDAN
    sys.modules.setdefault("PyEMD", pyemd_module)
