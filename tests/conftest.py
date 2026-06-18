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

    sift_module.sift = sift
    emd_module.sift = sift_module
    sys.modules.setdefault("emd", emd_module)
    sys.modules.setdefault("emd.sift", sift_module)

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
